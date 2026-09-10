import importlib.util
from pathlib import Path

import pytest

APP = "directions"
ENV = "ci"
ENDPOINT_NAME = "directions-runner"
API_KEY = "k"
APP_ID = "app1"

_SPEC = importlib.util.spec_from_file_location(
    "runpod_cleanup", Path(__file__).resolve().parent.parent / "scripts" / "runpod_cleanup.py"
)
runpod_cleanup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runpod_cleanup)


def _patch_graphql(monkeypatch: pytest.MonkeyPatch, app_data, env_data) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []

    def fake_graphql(query, variables, api_key):
        calls.append((query, variables))
        if query is runpod_cleanup.APP_QUERY:
            data = {"flashAppByName": app_data}
        elif query is runpod_cleanup.ENV_QUERY:
            data = {"flashEnvironmentByName": env_data}
        else:
            data = {"deleteEndpoint": None}
        (value,) = data.values()
        if isinstance(value, Exception):
            raise value
        return data

    monkeypatch.setattr(runpod_cleanup, "_graphql", fake_graphql)
    return calls


def _deletes(calls) -> list[str]:
    return [v["id"] for q, v in calls if q is runpod_cleanup.DELETE_MUTATION]


def test_delete_endpoints_deletes_only_matching_names(monkeypatch: pytest.MonkeyPatch) -> None:
    env_data = {"id": "env1", "endpoints": [{"id": "e1", "name": ENDPOINT_NAME}, {"id": "e2", "name": "other"}]}
    calls = _patch_graphql(monkeypatch, {"id": APP_ID}, env_data)

    deleted = runpod_cleanup.delete_endpoints(APP, ENV, ENDPOINT_NAME, API_KEY)

    assert deleted == ["e1"]
    assert _deletes(calls) == ["e1"]


def test_delete_endpoints_scopes_lookup_to_app_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_graphql(monkeypatch, {"id": APP_ID}, {"id": "env1", "endpoints": []})

    runpod_cleanup.delete_endpoints(APP, ENV, ENDPOINT_NAME, API_KEY)

    env_variables = next(v for q, v in calls if q is runpod_cleanup.ENV_QUERY)
    assert env_variables == {"input": {"flashAppId": APP_ID, "name": ENV}}


def test_delete_endpoints_missing_app_deletes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_graphql(monkeypatch, None, None)

    assert runpod_cleanup.delete_endpoints(APP, ENV, ENDPOINT_NAME, API_KEY) == []
    assert _deletes(calls) == []


def test_delete_endpoints_missing_env_deletes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_graphql(monkeypatch, {"id": APP_ID}, None)

    assert runpod_cleanup.delete_endpoints(APP, ENV, ENDPOINT_NAME, API_KEY) == []
    assert _deletes(calls) == []


def test_delete_endpoints_treats_not_found_error_as_nothing_to_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    not_found = runpod_cleanup.GraphQLError(f"GraphQL errors: Flash environment {ENV} in app {APP_ID} not found")
    calls = _patch_graphql(monkeypatch, {"id": APP_ID}, not_found)

    assert runpod_cleanup.delete_endpoints(APP, ENV, ENDPOINT_NAME, API_KEY) == []
    assert _deletes(calls) == []


def test_delete_endpoints_propagates_other_graphql_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_graphql(monkeypatch, {"id": APP_ID}, runpod_cleanup.GraphQLError("GraphQL errors: unauthorized"))

    with pytest.raises(runpod_cleanup.GraphQLError, match="unauthorized"):
        runpod_cleanup.delete_endpoints(APP, ENV, ENDPOINT_NAME, API_KEY)


def test_graphql_sets_a_user_agent_and_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"data": {"ok": true}}'

    def fake_urlopen(req, timeout):
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        return FakeResponse()

    monkeypatch.setattr(runpod_cleanup.urllib.request, "urlopen", fake_urlopen)

    assert runpod_cleanup._graphql("{ x }", {}, API_KEY) == {"ok": True}
    assert seen["headers"]["user-agent"] == runpod_cleanup.USER_AGENT
    assert seen["headers"]["authorization"] == f"Bearer {API_KEY}"


def test_main_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert runpod_cleanup.main(["--app", APP, "--env", ENV, "--endpoint-name", ENDPOINT_NAME]) == 1


def test_main_succeeds_with_nothing_to_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", API_KEY)
    _patch_graphql(monkeypatch, None, None)
    assert runpod_cleanup.main(["--app", APP, "--env", ENV, "--endpoint-name", ENDPOINT_NAME]) == 0


ENDPOINTS = [{"id": "e1", "name": "directions-runner"}, {"id": "e2", "name": "other"}]


def test_endpoint_exists_matches_id_and_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runpod_cleanup, "find_endpoints", lambda app, env, key: ENDPOINTS)
    assert runpod_cleanup.endpoint_exists("directions", "ci", "directions-runner", "e1", "k")
    assert not runpod_cleanup.endpoint_exists("directions", "ci", "directions-runner", "e2", "k")  # other name
    assert not runpod_cleanup.endpoint_exists("directions", "ci", "directions-runner", "e3", "k")


def test_main_exists_mode_deletes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr(runpod_cleanup, "find_endpoints", lambda app, env, key: ENDPOINTS)
    monkeypatch.setattr(runpod_cleanup, "delete_endpoints", lambda *a: pytest.fail("deleted in --exists mode"))
    args = ["--app", "directions", "--env", "ci", "--endpoint-name", "directions-runner"]
    assert runpod_cleanup.main([*args, "--exists", "e1"]) == 0
    assert runpod_cleanup.main([*args, "--exists", "e3"]) == 3

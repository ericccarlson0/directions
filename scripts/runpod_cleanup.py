#!/usr/bin/env python
"""Delete the endpoint a previous ``flash deploy`` left in a Flash environment.

``flash deploy`` decides whether an endpoint exists from ``.flash/resources.pkl``, which a fresh CI runner does
not have; a second deploy would attempt to re-create the endpoint and RunPod would reject the duplicate template
name. Deleting the previous endpoint first fixes this.

Usage::

    uv run scripts/runpod_cleanup.py --app directions --env ci --endpoint-name directions-runner
    uv run scripts/runpod_cleanup.py --app directions --env ci --endpoint-name directions-runner --exists <id>

The second form deletes nothing and reports whether an endpoint id is still live (the workflow reuses a recorded
endpoint when its handler digest is unchanged). Reads ``RUNPOD_API_KEY`` from the env. A missing app or env (the
API reports these as GraphQL "not found" errors, e.g. on the first deploy to a new env) means nothing to delete.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

GRAPHQL_URL = "https://api.runpod.io/graphql"
USER_AGENT = "directions-runpod-cleanup/1.0"  # the default urllib agent is rejected with HTTP 403

APP_QUERY = """
query getFlashAppByName($flashAppName: String!) {
  flashAppByName(flashAppName: $flashAppName) { id }
}
"""

ENV_QUERY = """
query getFlashEnvironmentByName($input: FlashEnvironmentByNameInput!) {
  flashEnvironmentByName(input: $input) { id endpoints { id name } }
}
"""

DELETE_MUTATION = """
mutation deleteEndpoint($id: String!) {
  deleteEndpoint(id: $id)
}
"""


class GraphQLError(RuntimeError):
    pass


def _graphql(query: str, variables: dict, api_key: str) -> dict:
    req = urllib.request.Request(
        GRAPHQL_URL,
        data=json.dumps({"query": query, "variables": variables}).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read().decode())
    if body.get("errors"):
        raise GraphQLError("GraphQL errors: " + "; ".join(e.get("message", str(e)) for e in body["errors"]))
    return body.get("data") or {}


def _lookup(query: str, variables: dict, api_key: str, key: str) -> dict | None:
    """An object lookup; ``None`` when the API reports it as not found."""
    try:
        return _graphql(query, variables, api_key).get(key)
    except GraphQLError as exc:
        if "not found" in str(exc).lower():
            return None
        raise


def find_endpoints(app: str, env: str, api_key: str) -> list[dict]:
    """Endpoints ``{id, name}`` registered in ``env`` of ``app``; empty if either does not exist."""
    app_data = _lookup(APP_QUERY, {"flashAppName": app}, api_key, "flashAppByName")
    if not app_data:
        return []
    env_data = _lookup(ENV_QUERY, {"input": {"flashAppId": app_data["id"], "name": env}}, api_key, "flashEnvironmentByName")
    if not env_data:
        return []
    return env_data.get("endpoints") or []


def delete_endpoints(app: str, env: str, endpoint_name: str, api_key: str) -> list[str]:
    """Delete every endpoint with name ``endpoint_name`` in the environment; return the deleted ids."""
    deleted = []
    for endpoint in find_endpoints(app, env, api_key):
        if endpoint.get("name") != endpoint_name:
            continue
        _graphql(DELETE_MUTATION, {"id": endpoint["id"]}, api_key)
        deleted.append(endpoint["id"])
    return deleted


def endpoint_exists(app: str, env: str, endpoint_name: str, endpoint_id: str, api_key: str) -> bool:
    """Whether the environment still has the endpoint ``endpoint_id`` under ``endpoint_name``."""
    return any(
        e.get("id") == endpoint_id and e.get("name") == endpoint_name for e in find_endpoints(app, env, api_key)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--exists", metavar="ENDPOINT_ID", default=None,
                        help="delete nothing; exit 0 if this endpoint id is live in the environment, else 3")
    args = parser.parse_args(argv)

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("RUNPOD_API_KEY is not set", file=sys.stderr)
        return 1

    if args.exists:
        if endpoint_exists(args.app, args.env, args.endpoint_name, args.exists, api_key):
            print(f"endpoint {args.exists} ({args.endpoint_name!r}) is live in {args.app}/{args.env}")
            return 0
        print(f"endpoint {args.exists} ({args.endpoint_name!r}) is not in {args.app}/{args.env}")
        return 3

    deleted = delete_endpoints(args.app, args.env, args.endpoint_name, api_key)
    if deleted:
        print(f"deleted endpoint(s) {', '.join(deleted)} named {args.endpoint_name!r} in {args.app}/{args.env}")
    else:
        print(f"no endpoint named {args.endpoint_name!r} in {args.app}/{args.env}; nothing to delete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

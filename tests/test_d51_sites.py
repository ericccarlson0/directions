"""scripts/d51_sites.py: the D51 site rule."""

import importlib.util
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location("d51_sites", Path(__file__).resolve().parents[1] / "scripts" / "d51_sites.py")
sites = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sites)


def test_sites_run_from_the_gather_to_where_the_answer_forms():
    assert sites.sites_for(24, 25) == [24, 25, 26]  # at least G..G+2
    assert sites.sites_for(24, 27) == [24, 25, 26, 27]  # F within G+3: every read point
    assert sites.sites_for(24, 33) == [24, 27, 30, 33]  # F beyond G+3: four, evenly spaced


def test_control_is_the_latest_quiet_read_point_two_below_the_gather():
    flips = {"S1": np.array([0, 0, 0.05, 0.3, 0.9, 1.0]), "S2": np.array([0, 0.02, 0.2, 0.5, 0.9, 1.0]),
             "S3": np.zeros(6)}
    assert sites.control_for(flips, 4) == 1  # read point 2 has S2 at 0.2
    assert sites.control_for({"S1": np.ones(6)}, 4) is None

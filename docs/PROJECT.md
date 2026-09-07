# Project: "Directions"

## Central Question

How does a low-dimensional causal control signal propagate through and get
transformed by successive layers of a neural network? How is control transformed into computation, and what is the difference, if any, between the two (is there any meaningful distinction, and if so, how is it measured)?

In particular, distinguish:

- conserved transmission (the control vector is maintained throughout layers)
- delayed activation (the conversion of control to computation is done at later layers; the control can stay dormant)
- cascaded activation (the conversion of control to computation is done gradually)
- amplification
- dimensional expansion
- creation of new, context-dependent computational directions

## Requirements

Results should be quantitative, reproducible, and maximally automatable.

## Distinguishing Modes

The pilot must reduce each task's layerwise profile to quantities that separate these "propagation modes", and emit them as a cross-task analysis. Operational definitions (measured downstream of the intervention layer, on held-out inputs):

| mode | signature |
|---|---|
| conserved transmission | control alignment \(A_l\) stays high; cumulative \(\log G\) ≈ 0; mean conversion \(C_l\) ≈ 0; \(d_{\mathrm{eff}}\) flat |
| delayed activation | conversion mass centred on late dominant block(s) with low early conversion |
| cascaded activation | conversion mass across blocks (entropy ratio near 1, no dominant block) |
| localized transformation | one block carries a large share of the conversion mass |
| amplification | cumulative \(\log G\) > 0 |
| dimensional expansion | \(d_{\mathrm{eff}}\) and \(d_{90}\) rise with depth; final/first ratio > 1 |
| new-direction creation | \(N_l\) high relative to matched random controls |

Note that **geometric** and **behavioural** localisation can disagree: conversion could be distributed across blocks while a single block carries most of the behavioural effect. Report the causal block ablation next to the geometry.
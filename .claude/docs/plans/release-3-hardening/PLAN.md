# 3.0 Release Hardening Implementation Plan

**Status:** Not started.

Seven PRs that take the unreleased 3.0 from "all refactors merged" to "safe to tag": a green master CI, one consistent `source -> target` orientation for every directed array (wrapper and `Connectivity` alike), actionable errors and a named `group_labels` argument at the front door, a README that leads with a working call, public simulators that the tutorials and tests share, planted-structure and cross-package validation for the measures that only had property tests, and a benchmark-gated speedup of the phase-lag-index family that dominates the default measure set. Line references were verified against master at commit `bcc16a8`; if the executor finds drift, re-grep the named symbol rather than trusting the number.

## Reading order

For agent invocation, **load only the slice you need**:

1. **Working a specific phase?** Open the matching phase file. Each phase file is self-contained: it lists upstream files to read, contracts/designs it depends on, tasks, validation slice, and fixtures.
2. **Need shared semantics?** [shared-contracts.md](shared-contracts.md).
3. **Need a per-component design?** [designs.md](designs.md).
4. **Need broader scope / risks / dependency policy?** [overview.md](overview.md).

## Files

- [overview.md](overview.md) — integration points, goals and non-goals, metrics, risks, rollout, open questions
- [shared-contracts.md](shared-contracts.md) — the orientation rule and the two public simulator signatures that phases 2, 5 and 6 share
- [designs.md](designs.md) — phase-lag moment restructuring (with measured prototype), `group_labels` routing, `sampling_frequency` validation, orientation swap sites
- Phases (each ships as a separable PR, in this order):
  - [phase-1-ci-zizmor.md](phase-1-ci-zizmor.md) — stop the zizmor SARIF upload that fails every master run
  - [phase-2-directed-orientation.md](phase-2-directed-orientation.md) — `Connectivity` directed arrays become `[..., source, target]`; delete the wrapper's transpose
  - [phase-3-input-validation-and-group-labels.md](phase-3-input-validation-and-group-labels.md) — typed `sampling_frequency` error, no 1000 Hz default, named `group_labels`, one kwargs story
  - [phase-4-docs-front-door.md](phase-4-docs-front-door.md) — README/docs index trim, DataArray cookbook recipe, tutorial opens with the wrapper
  - [phase-5-public-simulators.md](phase-5-public-simulators.md) — `simulate_lagged_broadband`, `simulate_shared_oscillation`; notebook rewritten on them with asserts
  - [phase-6-validation-gaps.md](phase-6-validation-gaps.md) — planted tests for CSD, MIC/MIM, blockwise Granger, partial coherence; nitime and mne-connectivity cross-checks
  - [phase-7-phase-lag-speedup.md](phase-7-phase-lag-speedup.md) — benchmark script, then the moment computation restructured under a 1e-12 parity gate

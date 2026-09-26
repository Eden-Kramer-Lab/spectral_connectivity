# Phase 1 — Make master CI green: stop the zizmor SARIF upload

[← back to PLAN.md](PLAN.md) · [overview](overview.md)

**Inputs to read first:**

- [.github/workflows/release.yml:18-58](../../../../.github/workflows/release.yml#L18-L58) — the workflow grants only `contents: read` (`:20-21`) and the `Code Quality` job's last step (`:55-58`) is `zizmorcore/zizmor-action` pinned at `cc914d7f…` (v0.6.4) with `persona: regular`.
- Evidence gathered on 2026-09-26: every master run since `chore(deps): bump the actions group` (#81, 2026-09-25) failed only in `Code Quality`; the job log shows zizmor itself exiting 0 with zero findings, then `github/codeql-action/upload-sarif` failing with `Resource not accessible by integration` because the token lacks `security-events: write`. `Build distribution`, `Test … install`, `Publish to PyPI` and `Create GitHub Release` were all skipped as a consequence, so a 3.0 tag today would not publish.
- The pinned action's `action.yml` has an input `advanced-security` (default `"true"`) that gates the upload step (`if: ${{ inputs.advanced-security == 'true' }}`), and an `annotations` input (default `"false"`) that is only allowed when `advanced-security` is `false`.

## Tasks

- In `.github/workflows/release.yml:55-58`, add two inputs to the zizmor step so findings surface as workflow annotations instead of a code-scanning upload:

  ```yaml
      - name: Lint the workflow itself
        uses: zizmorcore/zizmor-action@cc914d7f3750a2d13d75c7f184a1060aa0e9d482  # v0.6.4
        with:
          persona: regular
          # The repository has no GitHub code scanning, so the SARIF upload the
          # action does by default fails on `security-events`; report inline.
          advanced-security: false
          annotations: true
  ```

  Do not grant `security-events: write` instead: the repository's `security_and_analysis` settings show no code scanning, so the upload would still be refused, and the workflow's read-only default follows Scientific Python SPEC 8 on purpose (`:18-19`).
- `CHANGELOG.md`: no entry (CI-only change).

## Deliberately not in this phase

- Anything under `src/` or `tests/`.
- Re-enabling code scanning on the repository (a GitHub settings decision, not a workflow one).

## Validation slice

| Test | Asserts |
| --- | --- |
| PR run of `Code Quality` | job conclusion `success`; the zizmor step's log shows the audit result with no upload step executed |
| Next master push (after merge) | `gh run view <id> --json jobs` shows `Build distribution` and `Test … install` no longer `skipped` (they run on every push per the existing `needs`) |
| `uvx zizmor .github/workflows/release.yml` locally | no findings at the `regular` persona |

## Fixtures

None.

## Review

Before opening the PR for this phase, dispatch `code-reviewer` (or equivalent independent reviewer) against the diff. Confirm:
- Every task in this phase is implemented as specified.
- The "Deliberately not in this phase" list is honored — no scope creep into adjacent phases.
- Validation slice tests pass; slow / integration tests are marked.
- Tests aren't trivial — they exercise the asserted behavior, not tautologies (no `assert True`; no assertions that only verify the mock the test just configured). Shared setup is in fixtures, not copy-pasted across tests. (`testing-anti-patterns` covers the failure modes in detail.)
- Docstrings, test names, and module names don't reference this plan or its milestones.
- Old code paths flagged for removal in this phase are actually removed (no orphans left behind).
- User-facing documentation listed as tasks is updated, not deferred.

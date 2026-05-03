# Release Checklist

Use this checklist every time you make changes and want to publish a new
version of `rapidtextprep` to PyPI.

## 1. Make Your Code Changes

Edit the source code, tests, README, or other project files as needed.

Before publishing, make sure the package version is still unique. PyPI does not
allow uploading the same version twice.

## 2. Update the Version

Open `pyproject.toml` and update:

```toml
version = "0.1.1"
```

For the next release, increase it, for example:

```toml
version = "0.1.2"
```

Use simple semantic versioning:

- Patch release: `0.1.1` to `0.1.2` for fixes and small improvements.
- Minor release: `0.1.2` to `0.2.0` for new features.
- Major release: `0.2.0` to `1.0.0` for stable public API or breaking changes.

## 3. Update the Lockfile

Run:

```powershell
uv lock
```

This updates `uv.lock` so it matches `pyproject.toml`.

Important: if you forget this step, GitHub Actions may fail at:

```bash
uv sync --locked
```

## 4. Install and Sync Dependencies

Run:

```powershell
uv sync --locked
```

This confirms the lockfile is valid and installable.

## 5. Format, Lint, and Test

Run:

```powershell
uv run ruff format .
uv run ruff check .
uv run pytest
```

All checks should pass before you push.

## 6. Build the Package Locally

Run:

```powershell
uv build
```

This should create files like:

```text
dist/rapidtextprep-0.1.2.tar.gz
dist/rapidtextprep-0.1.2-py3-none-any.whl
```

The version number should match `pyproject.toml`.

## 7. Optional: Run the Benchmark

For normal cleaning:

```powershell
uv run python benchmarks/benchmark_pipeline.py --rows 100000 --chunk-size 20000 --n-jobs 5 --backend thread
```

For process-based cleaning:

```powershell
uv run python benchmarks/benchmark_pipeline.py --rows 100000 --chunk-size 20000 --n-jobs 5 --backend process
```

For lemmatization:

```powershell
uv run python benchmarks/benchmark_pipeline.py --rows 100000 --chunk-size 20000 --n-jobs 5 --backend thread --lemmatize
uv run python benchmarks/benchmark_pipeline.py --rows 100000 --chunk-size 20000 --n-jobs 5 --backend process --lemmatize
```

## 8. Check Git Status

Run:

```powershell
git status
```

Review the changed files. You should usually commit:

- Source files in `src/`
- Tests in `tests/`
- `README.md`
- `pyproject.toml`
- `uv.lock`
- Workflow or benchmark files if changed

Do not commit generated local folders such as:

- `.venv/`
- `.pytest_cache/`
- `.ruff_cache/`
- `__pycache__/`
- `dist/`

These are ignored by `.gitignore`.

## 9. Commit the Changes

Run:

```powershell
git add .
git commit -m "Release 0.1.2"
```

Use the actual version number in the commit message.

## 10. Push to GitHub

Run:

```powershell
git push
```

Confirm the latest commit appears on GitHub:

```text
https://github.com/suraj-yadav-aiml/rapidtextprep
```

## 11. Create a GitHub Release

Open:

```text
https://github.com/suraj-yadav-aiml/rapidtextprep/releases
```

Click:

```text
Draft a new release
```

Create a new tag using the same version as `pyproject.toml`, prefixed with `v`.

Example:

```text
v0.1.2
```

Release title:

```text
v0.1.2
```

Example release notes:

```markdown
## Changes

- Describe the main changes here.
- Mention bug fixes, new APIs, or performance improvements.
- Mention documentation updates if relevant.
```

Click:

```text
Publish release
```

## 12. Confirm GitHub Actions

After publishing the release, go to:

```text
https://github.com/suraj-yadav-aiml/rapidtextprep/actions
```

Open the `Publish to PyPI` workflow.

Confirm all steps pass:

- Install dependencies
- Run formatting check
- Run lint checks
- Run tests
- Build distributions
- Publish to PyPI

If the workflow fails with a lockfile error, run locally:

```powershell
uv lock
git add uv.lock
git commit -m "Update uv lock"
git push
```

Then run the workflow manually from the `main` branch.

## 13. Verify on PyPI

Open:

```text
https://pypi.org/project/rapidtextprep/
```

Confirm the new version is visible.

## 14. Test Install from PyPI

In a fresh environment, run:

```powershell
pip install rapidtextprep==0.1.2
```

Or with `uv`:

```powershell
uv pip install rapidtextprep==0.1.2
```

Then verify import:

```powershell
python -c "from rapidtextprep import clean_text; print(clean_text('I CAN''T wait!!!'))"
```

## Quick Command Summary

Replace `0.1.2` with the version you are releasing.

```powershell
uv lock
uv sync --locked
uv run ruff format .
uv run ruff check .
uv run pytest
uv build
git status
git add .
git commit -m "Release 0.1.2"
git push
```

Then create a GitHub Release named:

```text
v0.1.2
```

## Common Mistakes

- Forgetting to update `uv.lock` after changing `pyproject.toml`.
- Trying to publish a version that already exists on PyPI.
- Creating a GitHub tag before pushing the version bump.
- Rerunning a failed release workflow from an old tag commit.
- Forgetting to check the GitHub Actions logs after publishing a release.

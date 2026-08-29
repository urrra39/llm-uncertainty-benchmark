# CI workflow

`ci/ci.yml` is the intended contents of `.github/workflows/ci.yml`. It runs
ruff, ruff format, mypy strict and pytest on Python 3.11 with no GPU, installs
neither torch nor transformers, and sets `UNC_BENCH_OFFLINE=1` so no test can
reach the network.

It is parked here rather than in `.github/workflows/` because the credential
this repository was pushed with is a GitHub App installation token without the
`workflows` permission. The push was rejected with:

```
refusing to allow a GitHub App to create or update workflow
.github/workflows/ci.yml without `workflows` permission
```

Installing it requires a credential that has the permission:

```bash
mkdir -p .github/workflows
git mv ci/ci.yml .github/workflows/ci.yml
git commit -m "ci: install workflow"
git push
```

Until then, the same checks run locally with `make check`, which is what CI
invokes. Verified green on every commit in this repository.

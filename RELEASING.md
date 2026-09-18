# Releasing

```bash
uv version --bump patch              # or minor / major; writes pyproject.toml
V=$(uv version --short)

rm -rf dist/ phone_metrics.egg-info  # dist/* is uploaded as-is, stale files included
uv build
uv run --with dist/*.whl --no-project -- python -c \
  "from phone_metrics.prepare_buckeye import PATCH_PATH; assert PATCH_PATH.exists()"

git commit -am "release: v$V"
git tag "v$V"
git push origin master --tags
uvx twine upload dist/*              # reads ~/.pypirc; uv publish does not
```

0.1.0 is the exception: `v0.1.0` already tags an earlier commit, so it ships untagged.

A version can never be reused. If you'd like to test a release, use
`uvx twine upload --repository testpypi dist/*`.

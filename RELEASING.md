# Releasing

Make sure the release tag matches `version` in `pyproject.toml`.

After updating  `version`:

```bash
rm -rf dist/                        # uv publish uploads dist/*, stale files included
uv build
uv run --with dist/*.whl --no-project -- python -c \
  "from phone_metrics.prepare_buckeye import PATCH_PATH; assert PATCH_PATH.exists()"

git commit -am "release: v0.2.0"
git tag v0.2.0
git push origin master --tags
uv publish                          # token in ~/.pypirc or UV_PUBLISH_TOKEN
```

A version can never be reused. If you'd like to test a release, use
`uv publish --publish-url https://test.pypi.org/legacy/`.

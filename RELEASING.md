# Release Process

This repository follows Semantic Versioning.

1. Update CHANGELOG.md
2. Create release tag
3. Create a new release in GitHub
4. Notify team

## Release Steps in Poetry

Check the version in pyproject.toml

```bash
poetry version
```

Update to the next version

**Note** - do **not** include the 'v' in pyproject.toml. This may cause tag has no parent issues in projects that reference the code.

```bash
poetry version 0.1.1
```

On the main branch, tag the latest commit as the new release

```bash
git tag -a v0.1.1 -m "Fix important bug in logic v0.1.1"
```

Push the tag to GitHub

```bash
git push origin v0.1.1
```

List the tags locally

```bash
git tag
```

## Make Release in GitHub

Under the releases section there is an option to create a new release

## Additional Information

See [CONTRIBUTING.md](CONTRIBUTING.md) for additional guidelines

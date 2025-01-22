# Contributing

We welcome contributions of all forms to Mici.
All contributors are expected to abide by the [code of conduct](CODE_OF_CONDUCT.md).

If you don't have time to contribute but would like to show your support,
[starring](https://github.com/matt-graham/mici/stargazers) the repository or posting about it on social media is always appreciated!

## Reporting a bug or making a feature request

If you have a question please first check if it is covered in the [documentation](https://matt-graham.github.io/mici/)
or if there is an [existing issue](https://github.com/matt-graham/mici/issues) which answers your query.

If there is not a relevant existing issue, to report a problem you are having with the package
or request a new feature please [raise an issue](https://github.com/matt-graham/mici/issues/new/choose).

When reporting a bug, please described the expected behaviour and what you actual observe,
and sufficient information for someone else to _reproduce_ the problem. Ideally this
should be in the form of a [_minimal reproducible example_](https://en.wikipedia.org/wiki/Minimal_reproducible_example)
which reproduces the error while as being as small and simple as possible.

## Making a contribution

We use a _fork_ and _pull-request_ model for external contributions.
Before opening a pull request that proposes substantial changes to the repository,
for example adding a new feature or changing the public interface of the package,
please first [raise an issue](https://github.com/matt-graham/mici/issues/new/choose) outlining the problem the proposed changes would address
to allow for some discussion about the problem and proposed solution before significant
time is invested.

If you have not made an open-source contribution via a pull request before you may find
this [detailed guide](https://www.asmeurer.com/git-workflow/) by [asmeurer](https://github.com/asmeurer)
helpful. A summary of the main steps is as follows:

1. [Fork the repository](https://github.com/matt-graham/mici/fork) and create a local clone of the fork.
2. Create a new branch with a descriptive name in your fork clone.
3. Make the proposed changes on the branch, giving each commit a descriptive commit message.
4. Push the changes on the local branch to your fork on GitHub.
5. Create a [pull request](https://github.com/matt-graham/mici/compare),
   specifying the fork branch as the source of the changes,
   giving the pull request a descriptive title and explaining what you are changing and why in the description.
   If the pull-request is resolving a specific issue,
   use [keywords](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/using-keywords-in-issues-and-pull-requests)
   to link the appropriate issue.
6. Make sure all automated status checks pass on the pull request.
7. Await a review on the changes by one of the project maintainers, and address any review comments.
8. Once all status checks pass and the changes have been approved by a maintainer the pull request can be merged.

## Code style and linting

Mici uses the [Black](https://black.readthedocs.io/en/stable/the_black_code_style/index.html) code style,
and use [Ruff](https://docs.astral.sh/ruff/) to lint (and autoformat) the code.

One way to ensure changes do not violate any of the rules Ruff checks for is to
[install pre-commit](https://pre-commit.com/#install)
and use it to install pre-commit hooks into your local clone of the repository
using the configuration in [`.pre-commit-config.yaml`](.pre-commit-config.yaml) by running

```bash
pre-commit install
```

from the root of the repository.

This will run a series of checks on any staged changes when attempting to make a commit,
and flag if there are any problems with the changes. In some cases the problems may be
automatically fixed where possible; in this case the updated changes will need to be
staged and committed again.

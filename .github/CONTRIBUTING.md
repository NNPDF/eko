# How to Contribute

:tada: Thanks, for considering to contribute to EKO!

:pray: For the sake of simplicity we switch below to imperative
language, however, please read a "Please" in front of everything.

- :brain: Be reasonable and use common sense when contributing: we
  added some points we would like to highlight below
- :family: Follow our [Code of
  Conduct](https://github.com/N3PDF/eko/blob/master/.github/CODE_OF_CONDUCT.md)
  and use the provided [Issue
  Templates](https://github.com/N3PDF/eko/issues/new/choose)

## Tools

- :books: [`poetry`](https://github.com/python-poetry/poetry) is the
  dependency manager and packaging back-end of choice for this
  project - see the official [installation
  guide](https://python-poetry.org/docs/#installation)
- :hash: [`poery-dynamic-versioning`](https://github.com/mtkennerly/poetry-dynamic-versioning),
  is used to update the package version based on VCS status (tags and
  commits); note that since the version is dumped in output object,
  this is to be used not only for releases, but whenever output is
  generated (and intended to be used)
- :parking: [`pre-commit`](https://pre-commit.com/) is used to enforce
  automation and standardize the tools for all developers; if you want
  to contribute to this project, install it and setup

## Docs

- in order to run the notebooks in the environment, you need first to install
  the environment kernel; thus, run from inside the environment:
  ```sh
  # already installed with poetry, but in case...
  pip install ipykernel
  python -m ipykernel install --user --name=<env-name>
  ```
  thanks to [Nikolai Janakiev](https://janakiev.com/blog/jupyter-virtual-envs/#add-virtual-environment-to-jupyter-notebook)

## Testing

- :elephant: Make sure to not break the old tests (unless there was a
  mistake)
- :hatching_chick: Write new tests for your new code - the coverage
  should be back to 100% if possible

## Style Conventions

### Python Styleguide

- :art: Run [black](https://github.com/psf/black) to style your code
- :blue_book: Use [numpy documentation
  guide](https://numpydoc.readthedocs.io/en/latest/format.html)

### Git

- :octocat: Make sure the commit message is written properly ([This
  blogpost](https://chris.beams.io/posts/git-commit/) explains it
  nicely)
- :sailboat: Use [git-flow](https://github.com/nvie/gitflow) - note
  there is a [cheat
  sheet](https://danielkummer.github.io/git-flow-cheatsheet/index.html)
  and [shell
  completion](https://github.com/bobthecow/git-flow-completion)

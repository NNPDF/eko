<p align="center">
  <a href="https://n3pdf.github.io/eko/"><img alt="EKO" src="https://raw.githubusercontent.com/N3PDF/eko/master/doc/source/img/Logo.png" width=300></a>
</p>
<p align="center">
  <a href="https://github.com/N3PDF/eko/actions?query=workflow%3A%22eko%22"><img alt="Tests" src="https://github.com/N3PDF/eko/workflows/eko/badge.svg" /></a>
  <a href="https://eko.readthedocs.io/en/latest/?badge=latest"><img alt="Docs" src="https://readthedocs.org/projects/eko/badge/?version=latest"></a>
  <a href="https://codecov.io/gh/N3PDF/eko"><img src="https://codecov.io/gh/N3PDF/eko/branch/master/graph/badge.svg" /></a>
  <a href="https://www.codefactor.io/repository/github/n3pdf/eko"><img src="https://www.codefactor.io/repository/github/n3pdf/eko/badge" alt="CodeFactor" /></a>
</p>

EKO is a Python module to solve the DGLAP equations in terms of Evolution Kernel Operators in x-space.

## Installation
EKO is available via PyPI: <a href="https://pypi.org/project/eko/"><img alt="PyPI" src="https://img.shields.io/pypi/v/eko"/></a> - so you can simply run
```bash
pip install eko
```

### Development

If you want to install from source you can run
```bash
git clone git@github.com:N3PDF/eko.git
cd eko
poetry install
```

To setup `poetry`, and other tools, see [Contribution
Guidlines](https://github.com/N3PDF/eko/blob/master/.github/CONTRIBUTING.md).

## Documentation
- The documentation is available here: <a href="https://eko.readthedocs.io/en/latest/?badge=latest"><img alt="Docs" src="https://readthedocs.org/projects/eko/badge/?version=latest"></a>
- To build the documentation from source install [graphviz](https://www.graphviz.org/) and run in addition to the installation commands
```bash
pip install -r dev_requirements.txt
cd doc
make html
```

## Citation policy
Please cite our DOI when using our code: <a href="https://doi.org/10.5281/zenodo.3874237"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3874237.svg" alt="DOI"/></a>

## Contributing
- Your feedback is welcome! If you want to report a (possible) bug or want to ask for a new feature, please raise an issue: <a href="https://img.shields.io/github/issues/N3PDF/eko"><img alt="GitHub issues" src="https://img.shields.io/github/issues/N3PDF/eko"/></a>
- Please follow our [Code of Conduct](https://github.com/N3PDF/eko/blob/master/.github/CODE_OF_CONDUCT.md) and read the
  [Contribution Guidlines](https://github.com/N3PDF/eko/blob/master/.github/CONTRIBUTING.md)

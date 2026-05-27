<p align="center">
  <a href="https://eko.readthedocs.io/"><img alt="EKO" src="https://raw.githubusercontent.com/N3PDF/eko/master/doc/source/img/Logo.png" width=300></a>
</p>
<p align="center">
  <a href="https://github.com/N3PDF/eko/actions/workflows/unittests.yml"><img alt="Tests" src="https://github.com/N3PDF/eko/actions/workflows/unittests.yml/badge.svg" /></a>
  <a href="https://github.com/N3PDF/eko/actions/workflows/unittests-rust.yml"><img alt="Rust tests" src="https://github.com/N3PDF/eko/actions/workflows/unittests-rust.yml/badge.svg" /></a>
  <a href="https://eko.readthedocs.io/en/latest/?badge=latest"><img alt="Docs" src="https://readthedocs.org/projects/eko/badge/?version=latest"></a>
  <a href="https://codecov.io/gh/NNPDF/eko"><img src="https://codecov.io/gh/NNPDF/eko/branch/master/graph/badge.svg" /></a>
  <a href="https://www.codefactor.io/repository/github/nnpdf/eko"><img src="https://www.codefactor.io/repository/github/nnpdf/eko/badge" alt="CodeFactor" /></a>
</p>

EKO is a Python module to solve the DGLAP equations in N-space in terms of Evolution Kernel Operators in x-space.

## Installation
EKO is available via
- PyPI: <a href="https://pypi.org/project/eko/"><img alt="PyPI" src="https://img.shields.io/pypi/v/eko"/></a>: `$ pip install eko`
- conda-forge: [![Conda Version](https://img.shields.io/conda/vn/conda-forge/eko.svg)](https://anaconda.org/conda-forge/eko): `$ conda install eko`

The documentation is available here: <a href="https://eko.readthedocs.io/en/latest/?badge=latest"><img alt="Docs" src="https://readthedocs.org/projects/eko/badge/?version=latest"></a>

## ekore
We also provide a convenient access to the core elements of EKO: the anomalous dimensions $\gamma$ and operator matrix elements/transition matrix elements $A$.

These are collected from various references (see our documentation) and provide the current state of the art in one single place.
They mostly consist of (very) complicated experessions comprising many complicated math objects.

### Python

In Python you can access these elements through the `ekore` module installed together with the main Python library - [see our documentation](https://eko.readthedocs.io/en/latest/modules/ekore/ekore.html).

## Citation policy
When using our code please cite
- our DOI: <a href="https://doi.org/10.5281/zenodo.3874237"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.3874237.svg" alt="DOI"/></a>
- our paper: [![arXiv](https://img.shields.io/badge/arXiv-2202.02338-b31b1b?labelColor=222222)](https://arxiv.org/abs/2202.02338)

## Contributing
- Your feedback is welcome! If you want to report a (possible) bug or want to ask for a new feature, please raise an issue: <a href="https://github.com/N3PDF/eko/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/N3PDF/eko"/></a>
- If you need help, for installation, usage, or anything related, feel free to open a new discussion in the ["Support" section](https://github.com/NNPDF/eko/discussions/categories/support)
- Please follow our [Code of Conduct](https://github.com/N3PDF/eko/blob/master/.github/CODE_OF_CONDUCT.md) and read the
  [Contribution Guidelines](https://github.com/N3PDF/eko/blob/master/.github/CONTRIBUTING.md)

### Development installation

If you want to install from source you can run
```bash
git clone git@github.com:N3PDF/eko.git
cd eko
poetry install
```

To setup `poetry`, and other tools, see [Contribution
Guidelines](https://github.com/N3PDF/eko/blob/master/.github/CONTRIBUTING.md).


### Building the documentation
- The documentation is available here: <a href="https://eko.readthedocs.io/en/latest/?badge=latest"><img alt="Docs" src="https://readthedocs.org/projects/eko/badge/?version=latest"></a>
- To build the documentation from source install [graphviz](https://www.graphviz.org/) and run in addition to the installation commands
```bash
poe docs
```

### Tests and benchmarks
- To run unit test you can do
```bash
poe tests
```

- Benchmarks of specific part of the code, such as the strong coupling or msbar masses running, are available doing
```bash
poe bench
```

- The complete list of benchmarks with external codes is available through `ekomark`: [documentation](https://eko.readthedocs.io/en/latest/development/Benchmarks.html)


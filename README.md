# EXtra - European XFEL Toolkit for Research and Analysis

This repository holds various bits of code to make data analysis simpler.

## Development
During development in a custom environment it's often helpful to install the
package in editable mode:
```bash
$ pip install -e '.[tests,docs]'
```

Tests are stored under `tests/` and use pytest:
```bash
$ python -m pytest .
```

The documentation is written using [mkdocs](https://www.mkdocs.org/) with
[mkdocstrings-python](https://mkdocstrings.github.io/python/). To automatically
rebuild the docs while editing, run:
```bash
$ mkdocs serve
```

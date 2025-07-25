# EXtra - European XFEL Toolkit for Research and Analysis

[![codecov](https://codecov.io/gh/European-XFEL/EXtra/graph/badge.svg?token=vDu1zoDsEK)](https://codecov.io/gh/European-XFEL/EXtra)

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

## Deployment
The package is automatically deployed from `master` every night into the current
[`exfel-python`
environment](https://european-xfel.github.io/environments/environments/) using
the [`install-extra.sh`](docs/install-extra.sh) script (see the cron job with
`crontab -e` as `xsoft` on `max-exfl-display001.desy.de`).

If you want to update it manually, you must follow these steps:
1. SSH to Maxwell as the `xsoft` user
1. Run `./install-extra.sh`

Note: make sure to update the copy in `xsoft`'s home directory if the script is
updated:
```bash
rsync -a --progress docs/install-extra.sh xsoft@max-exfl-display.desy.de:/home/xsoft
```

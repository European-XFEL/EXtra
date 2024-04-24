# Accessing DAMNIT data

[DAMNIT](https://damnit.readthedocs.io) is a tool for automating analysis across
runs in a proposal, for the purpose of automatically building a run table and
doing intermediate analysis. It has a Python API to access the data it creates
which is accessible through `extra.damnit`, e.g.:
```python
from extra.damnit import Damnit
```

See the [API documentation](https://damnit.readthedocs.io/en/latest/api/) for
more information about how to use it.

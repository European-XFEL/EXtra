# Introduction
EXtra is a library developed at the European XFEL to provide a single entrypoint
for data analysis to users.

This is where you can find:

- Access to our other libraries for things like [reading data](reading-data.md)
  and handling [detector geometry](detector-geometry.md)
- Implementations of specific analysis techniques
- High-level components that abstract low-level Karabo devices
- Random useful things

## Installation
If you're using EXtra on the Maxwell cluster then you can access it through our
default environment (i.e. the `xfel (current)` kernel on Jupyterhub):
```bash
module load exfel exfel-python
```

This installation is always kept updated with the `master` branch. You can
also install it yourself with `pip`:
```bash
pip install euxfel-extra
```

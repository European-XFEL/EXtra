# Karabo bridge

The [Karabo
bridge](https://rtd.xfel.eu/docs/data-analysis-user-documentation/en/latest/online/#streaming-from-karabo-bridge)
is a protocol to stream data out of Karabo to a client. The Python client,
[`karabo-bridge-py`](https://github.com/European-XFEL/karabo-bridge-py), is
exposed under `extra.karabo_bridge`, e.g.:
```python
from extra.karabo_bridge import Client
```

See the [Karabo bridge
documentation](https://rtd.xfel.eu/docs/data-analysis-user-documentation/en/latest/software/karabo-bridge/#karabo-bridge)
for more details.

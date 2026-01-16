# Proposal API

This is an API for working with proposals. It lets you list runs, read metadata
from MyMDC, open data with EXtra-data, access DAMNIT results, and make simple
run timelines.

Quick example:
```python
from extra.proposal import Proposal

# Open a proposal
proposal = Proposal(1234)

# Print general information about the proposal
proposal.info()

# Get a table of samples and the runs they have
proposal.samples_table()

# Get DAQ data from extra-data
proposal.data()

# Get DAMNIT data through the DAMNIT API
proposal.damnit()
```

::: extra.proposal.Proposal
    options:
        filters:
        - "!^_.*"
        - "!.*search_source.*"

::: extra.proposal.RunReference

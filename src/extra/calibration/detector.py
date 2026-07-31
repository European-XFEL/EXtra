
import ast
import re
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatch
from string import Formatter

from .calcat import CalCatAPIError, get_client


class SourceExprChecker(ast.NodeVisitor):
    def visit_Call(self, node):
        raise ValueError("Function calls not allowed in source name patterns")


class SourceNameFormatter(Formatter):
    """String formatter that evaluates simple operations like {modno + 2}"""

    def get_field(self, field_name, args, kwargs):
        node = ast.parse(field_name, "<source pattern>", "eval")
        SourceExprChecker().visit(node)
        obj = eval(compile(node, "<source pattern>", "eval"), kwargs)
        return obj, 0


@dataclass
class DetectorModule:
    """Detector module.

    A module installed in a detector is represented by a physical
    detector unit (PDU) mapped to this module's logical position in the
    detector at a particular point in time.

    Calibration data is always associated with PDUs rather than the
    detector itself even in the case of a single module and thus PDU.

    Attributes:
        pdu_id (int): PDU numerical ID.
        physical_name (str): PDU identifier.
        aggregator (str): Data aggregator the PDU is mapped to.
        detector (str): Detector identifier the module is part of
        virtual_device_name (str): Identifier or QM name for the logical
            module within the detector, e.g. Q1M1.
        module_index (int): Enumerated module index within the detector
            when sorted by aggregator, contiguous and always starting at 0.
        module_number (int): Logical module number within the detector,
            may start at any number and have gaps.
        detector_type (str): Detector type string of this PDU.
    """

    pdu_id: int
    physical_name: str
    aggregator: str
    detector: str
    virtual_device_name: str | None
    module_index: int
    module_number: int | None
    detector_type: str
    legacy_uuid: int | None  # Deprecated, do not use
    source_name: str | None

    def __post_init__(self):
        if self.module_number is None:
            # Try to fill in module number if missing.
            self.module_number = int(re.findall(r"\d+", self.aggregator)[-1])

        if self.source_name is not None:
            self.source_name = SourceNameFormatter().format(
                self.source_name, modno=self.module_number)

    @property
    def ccv_params(self):
        """PDU arguments as needed for write_ccv()."""
        return self.physical_name, self.legacy_uuid, self.detector_type


class DetectorData(Mapping):
    """Detector consisting of one or more modules

    A detector can house one or more detector modules. For the purpose
    of tracking calibration data, a physical detector unit (PDU) is
    mapped to these detector modules. When a PDU is moved (i.e. mapped)
    to a different detector module or even a different detector,
    existing calibration data is therefore automatically applied
    correctly.

    This object exposes detector modules in a dict-like interface
    mapping data aggregators to detector modules at a particular point
    in time. Alternatively, the module index may also be used as a key.

    Some attributes may be `None` if this object was initialized from
    incomplete information rather than CalCat, e.g. metadata of
    correction jobs.

    Attributes:
        id (int): Detector numerical ID.
        identifier (str): Detector identifier.
        karabo_control_domain (str): Karabo domain for control devices.
        number_of_modules (int): Number of modules for the full detector,
            may be more than currently installed or selected.
        pdu_snapshot_at (str, optional): ISO format date the mapping is
            taken from.
    """

    def __init__(self, detector_row, module_rows_or_pdus, pdu_snapshot_at=None):
        # Result rows as returned by CalCat.

        self.id = detector_row['id']
        self.identifier = detector_row['identifier']
        self.karabo_control_domain = detector_row['karabo_id_control']
        self.number_of_modules = detector_row['number_of_modules']

        self._source_name_pattern = detector_row['source_name_pattern']
        self._first_module_index = detector_row['first_module_index']

        self.pdus = []

        def get_da(x):
            return x.aggregator if isinstance(x, DetectorModule) \
                else x['karabo_da']

        for i, item in enumerate(sorted(module_rows_or_pdus, key=get_da)):
            if not isinstance(item, DetectorModule):
                item = DetectorModule(
                    item['id'], item['physical_name'], item['karabo_da'],
                    self.identifier, item['virtual_device_name'], i,
                    item['module_number'], item['detector_type']['name'],
                    item['uuid'], self._source_name_pattern)
            else:
                item.module_index = i

            self.pdus.append(item)

        self.pdu_snapshot_at = pdu_snapshot_at

    @classmethod
    def _from_detector(cls, detector_row, pdu_snapshot_at, client):
        pdu_snapshot_at = client.format_time(pdu_snapshot_at)

        try:
            module_rows = client.get(
                'physical_detector_units/get_all_by_detector',
                {'detector_id': detector_row['id'],
                 'pdu_snapshot_at': client.format_time(pdu_snapshot_at)})
        except CalCatAPIError as e:
            if e.status_code == 404:
                module_rows = []
            else:
                raise e

        return cls(detector_row, module_rows, pdu_snapshot_at)

    @classmethod
    def from_numeric_id(cls, detector_id, pdu_snapshot_at=None, client=None):
        """Look up a detector and its modules by CalCat numeric ID.

        `pdu_snapshot_at` should either be an ISO 8601 compatible string
        or a datetime-like object. It may also be a DataCollection
        object from EXtra-data to use the beginning of the run as a
        point in time.
        """

        client = client or get_client()
        detector_row = client.detector_by_id(detector_id)

        return cls._from_detector(detector_row, pdu_snapshot_at, client)

    @classmethod
    def from_identifier(cls, identifier, pdu_snapshot_at=None, client=None):
        """Look up a detector and its modules by identifier.

        `pdu_snapshot_at` should either be an ISO 8601 compatible string
        or a datetime-like object. It may also be a DataCollection
        object from EXtra-data to use the beginning of the run as a
        point in time.
        """

        client = client or get_client()
        detector_row = client.detector_by_identifier(identifier)

        return cls._from_detector(detector_row, pdu_snapshot_at, client)

    @classmethod
    def from_instrument(cls, instrument, identifier=None, pdu_snapshot_at=None,
                        client=None):
        """Look up a detector and its modules by instrument.

        `identifier` may be a string restricting the result using Unix
        shell-style glob patterns.

        `pdu_snapshot_at` should either be an ISO 8601 compatible string
        or a datetime-like object. It may also be a DataCollection
        object from EXtra-data to use the beginning of the run as a
        point in time.
        """

        client = client or get_client()
        instrument_id = client.instrument_by_identifier(instrument)['id']

        rows = [det for det in client.get(
            'detectors/get_all_by_instrument', {'instrument_id': instrument_id}
        ) if identifier is None or fnmatch(det['identifier'], identifier)]

        if not rows:
            raise ValueError(f'No such detector found for {instrument}')
        elif len(rows) > 1:
            raise ValueError(
                f'Multiple such detectors found for {instrument}: ' +
                ', '.join([detector['identifier'] for detector in rows]))

        return cls._from_detector(rows[0], pdu_snapshot_at, client)

    @staticmethod
    def list_by_instrument(instrument, client=None):
        """List all detectors by instrument."""

        client = client or get_client()
        instrument_id = client.instrument_by_identifier(instrument)['id']

        return [det['identifier'] for det in
                client.get('detectors/get_all_by_instrument',
                           {'instrument_id': instrument_id})]

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.pdus[key]
        elif isinstance(key, str):
            for pdu in self.pdus:
                if pdu.aggregator == key:
                    return pdu

            raise KeyError(key)

    def __iter__(self):
        return (pdu.aggregator for pdu in self.pdus)

    def __len__(self):
        return len(self.pdus)

    def __repr__(self):
        return f'<DetectorData: {len(self.pdus)}/{self.number_of_modules} ' \
               f'modules of {self.identifier} on {self.pdu_snapshot_at}>'

    def _replace_modules(self, module_rows_or_pdus):
        detector_row = {'id': self.id, 'identifier': self.identifier,
                        'number_of_modules': self.number_of_modules,
                        'karabo_id_control': self.karabo_control_domain,
                        'source_name_pattern': self._source_name_pattern,
                        'first_module_index': self._first_module_index}

        return type(self)(detector_row, module_rows_or_pdus,
                          self.pdu_snapshot_at)

    @property
    def source_name_pattern(self) -> str:
        """Source name pattern."""
        assert self._source_name_pattern is not None, \
            'incomplete detector entry in CalCat'
        return self._source_name_pattern

    @property
    def source_names(self) -> str:
        """Source names."""
        assert self._source_name_pattern is not None, \
            'incomplete detector entry in CalCat'
        return [pdu.source_name for pdu in self.pdus]

    @property
    def first_module_index(self) -> int:
        """Module index of the first module."""
        assert self._first_module_index is not None, \
            'incomplete detector entry in CalCat'
        return self._first_module_index

    @property
    def detector_type(self) -> str:
        """Detector type of all PDUs if unique."""
        pdu_types = {pdu.detector_type for pdu in self.pdus}

        if len(pdu_types) > 1:
            # CalCat model enforces this.
            raise ValueError('more than one type of PDU: ' +
                             ', '.join(pdu_types))
        elif len(pdu_types) == 0:
            raise ValueError('no mapped PDUs')

        return pdu_types.pop()

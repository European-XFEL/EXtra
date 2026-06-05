
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from operator import index
from pathlib import Path
from typing import Dict, List, Optional, Union
from warnings import warn

import h5py
import pasha as psh

from extra_data.read_machinery import find_proposal
from ..data import DataCollection
from .calcat import get_client, get_default_caldb_root
from .detector import DetectorData
from .conditions import ConditionsBase, LPDConditions, detector_cond_cls


class NoDimensionLabelsError(ValueError):
    pass


def _summarise_mod_names(l):
    """Group module names like 'LPD03' into contiguous numeric ranges

    E.g. ['LPD00', 'LPD01', 'LPD03'] -> ['LPD00', 'LPD01'], ['LPD03']
    """
    grp = []
    num_re = re.compile(r'(\d+)')

    def extends_grp(split):
        if len(split) != len(grp[0]):
            return False

        diffs_at = [i for i, (p1, p2) in enumerate(zip(grp[-1], split))
                    if p1 != p2]
        if len(diffs_at) != 1 or ((diff_at := diffs_at[0]) % 2 == 0):
            return False  # >1 part changed, or non-numeric part changed

        if len(grp) > 2 and (grp[-1][diff_at] == grp[0][diff_at]):
            return False  # different part changing from current seq

        return int(grp[-1][diff_at]) + 1 == int(split[diff_at])

    for mod in l:
        parts = num_re.split(mod)  # parts 1, 3, ... are numeric

        if grp and not extends_grp(parts):
            yield [''.join(p) for p in grp]
            grp = [parts]
        else:
            grp.append(parts)

    if grp:
        yield [''.join(p) for p in grp]


@dataclass
class SingleConstant:
    """A calibration constant for one detector module

    CalCat calls this a calibration constant version (CCV).
    """

    path: Path
    dataset: str
    ccv_id: Optional[int]
    pdu_name: Optional[str]
    _metadata: dict = field(default_factory=dict)
    _have_calcat_metadata: bool = False

    @classmethod
    def from_response(cls, ccv: dict) -> "SingleConstant":
        return cls(
            path=Path(ccv["path_to_file"]) / ccv["file_name"],
            dataset=ccv["data_set_name"],
            ccv_id=ccv["id"],
            pdu_name=ccv["physical_detector_unit"]["physical_name"],
            _metadata=ccv,
            _have_calcat_metadata=True,
        )

    def file_path(self, caldb_root=None) -> Path:
        if caldb_root is not None:
            caldb_root = Path(caldb_root)
        else:
            caldb_root = get_default_caldb_root()
        return caldb_root / self.path

    def dataset_obj(self, caldb_root=None) -> h5py.Dataset:
        f = h5py.File(self.file_path(caldb_root), "r")
        return f[self.dataset]["data"]

    def ndarray(self, caldb_root=None):
        """Load the constant data as a Numpy array"""
        return self.dataset_obj(caldb_root)[:]

    def dimension_names(self, caldb_root=None):
        """Get the order of dimensions from the constant file (if it was saved)

        This is the same as the .dimensions property, but allows passing the
        root directory for calibration files.
        """
        try:
            return tuple(self.dataset_obj(caldb_root).attrs['dims'].tolist())
        except KeyError:
            raise NoDimensionLabelsError(
                "This constant was saved without dimension labels"
            )

    @property
    def dimensions(self):
        """Get the order of dimensions from the constant file (if it was saved)"""
        return self.dimension_names()

    def _load_calcat_metadata(self, client=None):
        client = client or get_client()
        calcat_meta = client.get(f"calibration_constant_versions/{self.ccv_id}")
        # Any metadata we already have takes precedence over CalCat, so
        # this can't change a value that was previously returned.
        self._metadata = calcat_meta | self._metadata
        self._have_calcat_metadata = True

    def metadata(self, key, client=None):
        """Get a specific metadata field, e.g. 'begin_validity_at'

        This may make a request to CalCat if the value is not already known.
        """
        if key not in self._metadata and not self._have_calcat_metadata:
            if self.ccv_id is None:
                raise KeyError(f"{key!r} (no CCV ID to request data from CalCat")
            self._load_calcat_metadata(client)

        return self._metadata[key]

    def metadata_dict(self, client=None):
        """Get a dict of available metadata

        If this constant didn't come from CalCat but we have a CalCat CCV ID,
        this will fetch metadata from CalCat.
        """
        if (not self._have_calcat_metadata) and (self.ccv_id is not None):
            self._load_calcat_metadata(client)
        return self._metadata.copy()


def prepare_selection(
        module_details, module_nums=None, aggregator_names=None, qm_names=None
):
    aggs = aggregator_names  # Shorter name -> fewer multi-line statements
    n_specified = sum([module_nums is not None, aggs is not None, qm_names is not None])
    if n_specified > 1:
        raise TypeError(
            "select_modules() accepts only one of module_nums, aggregator_names "
            "& qm_names"
        )

    if module_nums is not None:
        by_mod_no = {m["module_number"]: m for m in module_details}
        return [by_mod_no[n]["karabo_da"] for n in module_nums]
    elif qm_names is not None:
        by_qm = {m["virtual_device_name"]: m for m in module_details}
        return [by_qm[s]["karabo_da"] for s in qm_names]
    elif aggs is not None:
        miss = set(aggs) - {m["karabo_da"] for m in module_details}
        if miss:
            raise KeyError("Aggregators not found: " + ", ".join(sorted(miss)))
        return aggs
    else:
        raise TypeError("select_modules() requires an argument")


@dataclass
class MultiModuleConstant(Mapping):
    """A group of similar constants for several modules of one detector.

    This works as a mapping holding `SingleConstant` objects.
    Keys can be module numbers (`offset[0]`), data aggregator names
    (`offset['LPD00']`), QxMy names (`offset['Q1M1']`) or Physical Detector Unit
    (PDU) names.
    """

    constants: Dict[str, SingleConstant]  # Keys e.g. 'LPD00'
    module_details: List[Dict]
    detector_name: str  # e.g. 'HED_DET_AGIPD500K2G'
    calibration_name: str

    def __repr__(self):
        return (
            f"<MultiModuleConstant: {self.calibration_name} for "
            f"{len(self.constants)} modules of {self.detector_name}>"
        )

    def __iter__(self):
        return iter(self.constants)

    def __len__(self):
        return len(self.constants)

    def __getitem__(self, key):
        if key in (None, ""):
            raise KeyError(key)

        candidate_kdas = set()
        if key in self.constants:  # Karabo DA name, e.g. 'LPD00'
            candidate_kdas.add(key)

        undef = object()
        for m in self.module_details:
            names = (
                m.get("module_number", undef),
                m.get("virtual_device_name", undef),
                m["physical_name"]
            )
            if key in names and m["karabo_da"] in self.constants:
                candidate_kdas.add(m["karabo_da"])

        if not candidate_kdas:
            raise KeyError(key)
        elif len(candidate_kdas) > 1:
            raise KeyError(f"Ambiguous key: {key} matched {candidate_kdas}")

        return self.constants[candidate_kdas.pop()]

    def select_modules(
            self, module_nums=None, *, aggregator_names=None, qm_names=None
    ) -> "MultiModuleConstant":
        """Return a new `MultiModuleConstant` object with only the selected modules

        One of `module_nums`, `aggregator_names` or `qm_names` must be specified.
        """
        aggs = prepare_selection(
            self.module_details, module_nums, aggregator_names, qm_names
        )
        d = {aggr: scv for (aggr, scv) in self.constants.items() if aggr in aggs}
        mods = [m for m in self.module_details if m["karabo_da"] in d]
        return replace(self, constants=d, module_details=mods)

    # These properties label only the modules we have constants for, which may
    # be a subset of what's in module_details
    @property
    def aggregator_names(self):
        "Data aggregator names for the modules where we have this constant"
        return sorted(self.constants)

    @property
    def module_nums(self):
        "Module numbers for the modules where we have this constant"
        return [
            m["module_number"]
            for m in self.module_details
            if m["karabo_da"] in self.constants
        ]

    @property
    def qm_names(self):
        "Names like Q1M3 for the modules where we have this constant, if applicable"
        return [
            m["virtual_device_name"]
            for m in self.module_details
            if m["karabo_da"] in self.constants
        ]

    @property
    def pdu_names(self):
        """Names of the specific detector units making up the detector.

        Only includes modules where we have this constant."""
        return [
            m["physical_name"]
            for m in self.module_details
            if m["karabo_da"] in self.constants
        ]

    def ndarray(self, caldb_root=None, *, parallel=0):
        """Load this constant as a Numpy array.

        If `parallel` is specified, the per-module constants are loaded in
        parallel using N worker processes.
        """
        eg_dset = self.constants[self.aggregator_names[0]].dataset_obj(caldb_root)
        shape = (len(self.constants),) + eg_dset.shape

        if parallel > 0:
            load_ctx = psh.ProcessContext(num_workers=parallel)
        else:
            load_ctx = psh.SerialContext()

        arr = psh.alloc(shape, eg_dset.dtype, fill=0)

        def _load_constant_dataset(wid, index, mod):
            dset = self.constants[mod].dataset_obj(caldb_root)
            dset.read_direct(arr[index])

        load_ctx.map(_load_constant_dataset, self.aggregator_names)
        return arr

    def dimension_names(self, caldb_root=None):
        """Get the order of dimensions for this constant (if it was saved)

        Possible dimension names include "module", "cell", "gain", "fast_scan"
        and "slow_scan".

        This is the same as the .dimensions property, but allows passing the
        root directory for calibration files.
        """
        # We'll assume the constants for different modules have the same axis
        # order. The ndarray and xarray methods also assume this.
        kda = next(iter(self.constants))
        return ("module",) + self.constants[kda].dimension_names(caldb_root)

    @property
    def dimensions(self):
        """Get the order of dimensions for this constant (if it was saved)"""
        return self.dimension_names()

    def xarray(self, module_naming="modnum", caldb_root=None, *, parallel=0):
        """Load this constant as an xarray DataArray.

        `module_naming` may be "modnum", "aggregator" or "qm" to use different
        styles of labelling for the modules dimension.

        If `parallel` is specified, the per-module constants are loaded in
        parallel using N worker processes.
        """
        import xarray

        if module_naming == "aggregator":
            modules = self.aggregator_names
        elif module_naming == "modnum":
            modules = self.module_nums
        elif module_naming == "qm":
            modules = self.qm_names
        else:
            raise ValueError(
                f"{module_naming=} (must be 'aggregator', 'modnum' or 'qm')"
            )

        ndarr = self.ndarray(caldb_root, parallel=parallel)

        # Dimension labels
        try:
            dims = self.dimension_names(caldb_root)
        except NoDimensionLabelsError:
            dims = ["module"] + ["dim_%d" % i for i in range(ndarr.ndim - 1)]
        coords = {"module": modules}
        name = self.calibration_name

        return xarray.DataArray(ndarr, dims=dims, coords=coords, name=name)


class CalibrationData(Mapping):
    """Collected constants for a given detector

    This can represent multiple constant types (offset, gain, bad pixels, etc.)
    across multiple modules. It works as a mapping keyed by constant type
    (e.g. `cd["Offset"]`), giving you `MultiModuleConstant` objects.
    """

    def __init__(self, constant_groups, detector, condition=None):
        # {calibration: {karabo_da: SingleConstant}}
        self.constant_groups = constant_groups
        self.detector = detector
        self._condition = condition

    @staticmethod
    def _format_cond(condition):
        """Encode operating condition to CalCat API format.

        Args:
            condition (dict): Mapping of parameter DB name to value

        Returns:
            (dict) Operating condition for use in CalCat API.
        """

        if not all([isinstance(v, (float, str)) for v in condition.values()]):
            raise TypeError('Operating condition parameters may only be '
                            'float or str')

        return {
            "parameters_conditions_attributes": [{
                "parameter_name": k,
                "value": v
            } for k, v in condition.items()
            ]
        }

    @classmethod
    def from_condition(
            cls,
            condition: "ConditionsBase",
            detector_name,
            calibrations=None,
            client=None,
            event_at=None,
            pdu_snapshot_at=None,
            begin_at_strategy="closest",
    ):
        """Look up constants for the given detector conditions & timestamp.

        `condition` should be a conditions object for the relevant detector type,
        e.g. `DSSCConditions`.

        `event_at` and `pdu_snapshot_at` should either be an ISO 8601
        compatible string or a datetime-like object. It may also be a
        DataCollection object from EXtra-data to use the beginning of the
        run as a point in time.
        """
        accepted_strategies = ["closest", "prior"]
        if begin_at_strategy not in accepted_strategies:
            raise ValueError(
                "Invalid begin_at_strategy. "
                f"Expected one of {accepted_strategies}")

        if calibrations is None:
            calibrations = set(condition.calibration_types)
        if pdu_snapshot_at is None:
            pdu_snapshot_at = event_at

        cal_types_by_params_used = {}
        for cal_type, params in condition.calibration_types.items():
            if cal_type in calibrations:
                cal_types_by_params_used.setdefault(tuple(params), []).append(cal_type)

        client = client or get_client()

        detector = DetectorData.from_identifier(detector_name, pdu_snapshot_at=pdu_snapshot_at)

        constant_groups = {}

        for params, cal_types in cal_types_by_params_used.items():
            condition_dict = condition.make_dict(params)

            cal_id_map = {
                client.calibration_by_name(name)["id"]: name for name in cal_types
            }
            calibration_ids = list(cal_id_map.keys())

            query_res = client.get(
                "calibration_constant_versions/get_by_detector_conditions",
                {
                    "detector_identifier": detector_name,
                    "calibration_id": str(calibration_ids),
                    "karabo_da": "",
                    "event_at": client.format_time(event_at),
                    "pdu_snapshot_at": client.format_time(pdu_snapshot_at),
                    "begin_at_strategy": begin_at_strategy,
                },
                data=json.dumps(cls._format_cond(condition_dict)),
            )

            for ccv in query_res:
                aggr = ccv["physical_detector_unit"]["karabo_da"]
                cal_type = cal_id_map[ccv["calibration_constant"]["calibration_id"]]

                const_group = constant_groups.setdefault(cal_type, {})
                const_group[aggr] = SingleConstant.from_response(ccv)

        return cls(constant_groups, detector, condition)

    @classmethod
    def from_report(
            cls,
            report_id_or_path: Union[int, str],
            client=None,
    ):
        """Look up constants by a report ID or path.

        Constants produced together in the same characterisation are grouped
        in CalCat by their report. This method accepts either the integer report
        ID or the full filesystem path of the report.
        """
        client = client or get_client()

        # Use max page size, hopefully always enough for CCVs from 1 report
        params = {"page_size": 500}
        if isinstance(report_id_or_path, int):
            params["report_id"] = report_id_or_path  # Numeric ID
        else:
            params["report.file_path"] = str(report_id_or_path)

        res = client.get("calibration_constant_versions", params)

        constant_groups = {}
        pdus = {}  # keyed by karabo_da (e.g. 'AGIPD00')
        det_ids = set()  # Should only have one detector

        for ccv in res:
            pdu = ccv["physical_detector_unit"]
            # We're only interested in the PDU mapping from the CCV start time
            kda = pdu["karabo_da"] = pdu.pop("karabo_da_at_ccv_begin_at")
            det_id = pdu["detector_id"] = pdu.pop("detector_id_at_ccv_begin_at")
            pdu["virtual_device_name"] = pdu.pop("virtual_device_name_at_ccv_begin_at")
            if pdu.get("module_number_at_ccv_begin_at") is not None:
                pdu["module_number"] = pdu.pop("module_number_at_ccv_begin_at")
            else:
                pdu["module_number"] = int(re.findall(r"\d+", kda)[-1])

            det_ids.add(det_id)
            if kda in pdus:
                if pdu["physical_name"] != pdus[kda]["physical_name"]:
                    raise Exception(
                        f"Mismatched PDU mapping from calibration report: {kda} is both"
                        f" {pdu['physical_name']} and {pdus[kda]['physical_name']}"
                    )
            else:
                pdus[kda] = pdu

            cal_type = client.calibration_by_id(
                ccv["calibration_constant"]["calibration_id"]
            )["name"]
            const_group = constant_groups.setdefault(cal_type, {})
            const_group[kda] = SingleConstant.from_response(ccv)

        if len(det_ids) > 1:
            raise Exception(f"Found multiple detector IDs in report: {det_ids}")
        # The "identifier", "name" & "karabo_name" fields seem to have the same names

        detector_row = client.detector_by_id(det_ids.pop())
        detector_types = {det_type["id"]: det_type for det_type
                          in client.get("detector_types")}

        # Extend PDUs by data missing in the report result set.
        for pdu in pdus.values():
            pdu["detector"] = detector_row
            pdu["detector_type"] = detector_types[pdu["detector_type_id"]]

        return cls(constant_groups, DetectorData(
            detector_row, sorted(pdus.values(), key=lambda x: x["karabo_da"])))

    @staticmethod
    def _read_correction_file(metadata_path: Path):
        import yaml

        if metadata_path.is_dir():
            metadata_path = metadata_path / "calibration_metadata.yml"

        with metadata_path.open('r') as f:
            metadata = yaml.safe_load(f)

        constant_groups = {}  # keyed by calib type (e.g. 'Offset'), then karabo_da ('AGIPD00')
        pdus = {}  # keyed by karabo_da (e.g. 'AGIPD00')
        det_name = metadata['calibration-configurations']['karabo-id']

        for kda, grp in metadata['retrieved-constants'].items():
            if 'constants' not in grp:  # e.g. 'time-summary' key
                continue

            pdu_name = grp['physical-name']
            # Missing: module_number, virtual_device_name
            pdus[kda] = {'karabo_da': kda, 'physical_name': pdu_name}

            for cal_type, details in grp['constants'].items():
                const_group = constant_groups.setdefault(cal_type, {})
                const_group[kda] = SingleConstant(
                    path=details['path'],
                    dataset=details['dataset'],
                    ccv_id=details['ccv_id'],
                    pdu_name=pdu_name,
                    _metadata={'begin_validity_at': details['creation-time']}
                )

        return constant_groups, pdus, det_name

    @classmethod
    def from_correction(
            cls,  metadata_file_or_proposal: str | Path | int, run: int | None =None,
            detector_name=None, *, client=None, use_calcat=True
    ):
        """Find constants used to produce corrected data.

        This can be called with a proposal & run number and a detector name
        (e.g. 'FXE_XAD_JF1M'), or with the path of a YAML metadata file from the
        EuXFEL offline calibration pipeline.

        By default, this method retrieves additional metadata from CalCat.
        Pass use_calcat=False to read only the minimal info in a YAML file.
        """
        if run is not None:
            if detector_name is None:
                raise TypeError("detector_name required with proposal/run numbers")
            proposal = metadata_file_or_proposal
            if isinstance(proposal, str):
                if ('/' not in proposal) and not proposal.startswith('p'):
                    proposal = 'p' + proposal.rjust(6, '0')
            else:
                # Allow integers, including numpy integers
                proposal = 'p{:06d}'.format(index(proposal))

            run_dir = Path(find_proposal(proposal)) / 'proc' / f'r{run:04d}'
            yaml_path = run_dir / f"calibration_metadata_{detector_name}.yml"
        elif detector_name is not None:
            raise TypeError("detector_name was passed but run was not")
        else:
            yaml_path = Path(metadata_file_or_proposal)

        constant_groups, pdus, det_name = cls._read_correction_file(yaml_path)
        module_details = sorted(pdus.values(), key=lambda d: d["karabo_da"])

        if not use_calcat:
            detector_row = {
                'id': None, 'identifier': det_name,
                'karabo_id_control': None, 'number_of_modules': None,
                'source_name_pattern': None,  'first_module_index': None}

            for pdu in pdus.values():
                pdu.update(
                    id=None, uuid=None, detector={'identifier': det_name},
                    virtual_device_name=None, module_number=None,
                    detector_type={'name': None})

            return cls(
                constant_groups, DetectorData(detector_row, pdus.values()))

        # Get module_number, virtual_device_name from CCV info if possible
        need_metadata = {
            sc.ccv_id: sc for mmc in constant_groups.values() for sc in mmc.values()
        }

        def extend_module_info(pdu_dict):
            kda = pdu_dict['karabo_da_at_ccv_begin_at']

            pdus[kda].update(pdu_dict)
            pdus[kda]['karabo_da'] = pdu_dict['karabo_da_at_ccv_begin_at']

            if vdn := pdu_dict['virtual_device_name_at_ccv_begin_at']:
                pdus[kda]['virtual_device_name'] = vdn
            if modnum := pdu_dict['module_number_at_ccv_begin_at']:
                pdus[kda]['module_number'] = modnum

        # Retrieve constant metadata from CalCat
        # If possible, we retrieve batches of CCVs by report ID.
        # A correction will normally use e.g. offset constants for all
        # modules from one report, i.e. generated at the same time.
        # E.g. for LPD-1M, this can do 4 requests (2 CCVs + 2 reports)
        # instead of 96 individual CCVs.
        client = client or get_client()
        while need_metadata:
            # .metadata_dict() fetches & caches CCV metadata from CalCat
            md = need_metadata.popitem()[1].metadata_dict()

            extend_module_info(md['physical_detector_unit'])

            if (report_id := md['report_id']) is None:
                continue  # CCV not associated with a report

            # Fill metadata for other CCVs belonging to this report
            report_info = client.get(f"reports/{report_id}")
            for ccv_dict in report_info['calibration_constant_versions']:
                if const_obj := need_metadata.pop(ccv_dict['id'], None):
                    const_obj._metadata = ccv_dict
                    const_obj._have_calcat_metadata = True
                extend_module_info(ccv_dict['physical_detector_unit'])

        detector_row = client.detector_by_identifier(det_name)
        detector_types = {det_type["id"]: det_type for det_type
                          in client.get("detector_types")}

        # Extend PDUs by data missing in the report result set.
        for da, pdu in pdus.items():
            pdu["detector"] = detector_row
            pdu["detector_type"] = detector_types[pdu["detector_type_id"]]

        return cls(constant_groups, DetectorData(detector_row, pdus.values()))

    @classmethod
    def from_data(
            cls,
            data: 'DataCollection',
            detector_name: str,
            calibrations=None,
            client=None,
            begin_at_strategy='closest',
            **kwargs
    ):
        """Look up constants applicable to given a dataset.

        `data` should be an EXtra-data `DataCollection` object containing
        the necessary metadata to identify the detector conditions.
        `detector_name` refers to the detector identifer as used in CalCat,
        typically identical to its Karabo domain, i.e. the first part of
        its device IDs.

        The remaining arguments behave in the same way as for
        `CalibrationData.from_condition` and any additional keyword
        arguments are passed on to the applicable `ConditionsBase.from_data`
        method, e.g. `AGIPDConditions.from_data`.
        """

        creation_date = data[0].train_timestamps(pydatetime=True)[0]

        client = client or get_client()
        det = DetectorData.from_identifier(detector_name, client=client,
                                           pdu_snapshot_at=creation_date)

        try:
            cond_cls = detector_cond_cls[det.detector_type]
        except KeyError:
            raise NotImplementedError(det.detector_type)

        return cls.from_condition(
            cond_cls.from_data(data, detector_name, client=client, **kwargs),
            detector_name, calibrations=calibrations, client=client,
            event_at=creation_date, pdu_snapshot_at=creation_date,
            begin_at_strategy=begin_at_strategy)

    def __getitem__(self, key):
        if isinstance(key, str):
            return MultiModuleConstant(
                self.constant_groups[key], self.module_details, self.detector_name, key
            )
        elif isinstance(key, tuple) and len(key) == 2:
            cal_type, module = key
            return self[cal_type][module]
        else:
            raise TypeError(f"Key should be string or 2-tuple (got {key!r})")

    def __iter__(self):
        return iter(self.constant_groups)

    def __len__(self):
        return len(self.constant_groups)

    def __bool__(self):
        # Do we have any constants of any type?
        return any(bool(grp) for grp in self.constant_groups.values())

    def __contains__(self, item):
        return item in self.constant_groups

    def __repr__(self):
        return (
            f"<CalibrationData: {', '.join(sorted(self.constant_groups))} "
            f"constants for {len(self.module_details)} modules of {self.detector_name}>"
        )

    # These properties may include modules for which we have no constants -
    # when created with .from_condition(), they represent all modules present in
    # the detector (at the specified time).
    @property
    def module_nums(self):
        "Module numbers in the detector. May include missing modules."
        return [m["module_number"] for m in self.module_details]

    @property
    def aggregator_names(self):
        "Data aggregator names for modules. May include missing modules."
        return [m["karabo_da"] for m in self.module_details]

    @property
    def qm_names(self):
        "Module names like Q1M3, if present. May include missing modules."
        return [m["virtual_device_name"] for m in self.module_details]

    @property
    def pdu_names(self):
        """Names of the specific detector units making up the detector.

        May include missing modules."""
        return [m["physical_name"] for m in self.module_details]

    @property
    def module_details(self):
        return [dict(
            id=pdu.pdu_id, physical_name=pdu.physical_name,
            karabo_da=pdu.aggregator,
            virtual_device_name=pdu.virtual_device_name, uuid=pdu.legacy_uuid,
            module_number=pdu.module_number,
            detector_type=dict(name=pdu.detector_type)
        ) for pdu in self.detector.values()]

    @property
    def detector_name(self):
        return self.detector.identifier

    def require_calibrations(self, calibrations) -> "CalibrationData":
        """Drop any modules missing the specified constant types"""
        mods = set(self.aggregator_names)
        for cal_type in calibrations:
            if cal_type in self:
                mods.intersection_update(self[cal_type].constants)
            else:
                mods = set()  # None of this found
        return self.select_modules(aggregator_names=mods)

    def select_modules(
            self, module_nums=None, *, aggregator_names=None, qm_names=None
    ) -> "CalibrationData":
        """Return a new `CalibrationData` object with only the selected modules

        One of `module_nums`, `aggregator_names` or `qm_names` must be specified.
        """
        # Validate the specified modules against those we know about.
        # Each specific constant type may have only a subset of these modules.
        aggs = prepare_selection(
            self.module_details, module_nums, aggregator_names, qm_names
        )
        constant_groups = {}
        matched_aggregators = set()
        for cal_type, const_group in self.constant_groups.items():
            constant_groups[cal_type] = d = {
                aggr: const for (aggr, const) in const_group.items() if aggr in aggs
            }
            matched_aggregators.update(d.keys())

        module_details = [
            m for m in self.module_details if m["karabo_da"] in matched_aggregators
        ]

        return type(self)(
            constant_groups, self.detector._replace_modules(module_details))

    def select_calibrations(self, calibrations) -> "CalibrationData":
        """Return a new `CalibrationData` object with only the selected constant types"""
        const_groups = {c: self.constant_groups[c] for c in calibrations}
        return type(self)(const_groups, self.detector)

    def merge(self, *others: "CalibrationData") -> "CalibrationData":
        """Combine two or more `CalibrationData` objects for the same detector.

        Where the inputs have different constant types or different modules,
        the output will include all of them (set union). Where they overlap,
        later inputs override earlier ones.
        """
        det_names = set(cd.detector_name for cd in (self,) + others)
        if len(det_names) > 1:
            raise Exception(
                "Cannot merge calibration data for different "
                "detectors: " + ", ".join(sorted(det_names))
            )
        det_name = det_names.pop()

        cal_types = set(self.constant_groups)
        aggregators = set(self.aggregator_names)
        pdus_d = {m["karabo_da"]: m for m in self.module_details}
        for other in others:
            cal_types.update(other.constant_groups)
            aggregators.update(other.aggregator_names)
            for md in other.module_details:
                # Warn if constants don't refer to same modules
                md_da = md["karabo_da"]
                if md_da in pdus_d:
                    pdu_a = pdus_d[md_da]["physical_name"]
                    pdu_b = md["physical_name"]
                    if pdu_a != pdu_b:
                        warn(
                            f"Merging constants with different modules for "
                            f"{md_da}: {pdu_a!r} != {pdu_b!r}",
                            stacklevel=2,
                        )
                else:
                    pdus_d[md_da] = md

        module_details = sorted(pdus_d.values(), key=lambda d: d["karabo_da"])

        constant_groups = {}
        for cal_type in cal_types:
            d = constant_groups[cal_type] = {}
            for caldata in (self,) + others:
                if cal_type in caldata:
                    d.update(caldata.constant_groups[cal_type])

        return type(self)(
            constant_groups, self.detector._replace_modules(module_details))

    def summary_table(self, module_naming="modnum"):
        """Make a table overview of the constants found.

        Columns are calibration types, rows are modules.
        If there are >4 calibrations, the table will be split up into several
        pieces with up to 4 calibrations in each.

        The table(s) returned should be rendered within Jupyter notebooks,
        including when converting them to Latex & PDF.

        Args:
            module_naming (str): modnum, aggregator or qm, to change how the
                modules are labelled in the table. Defaults to modnum.
        """
        if module_naming == "aggregator":
            modules = self.aggregator_names
        elif module_naming == "modnum":
            modules = self.module_nums
        elif module_naming == "qm":
            modules = self.qm_names
        else:
            raise ValueError(
                f"{module_naming=} (must be 'aggregator', 'modnum' or 'qm')"
            )

        cal_groups = [
            sorted(self.constant_groups)[x:x+4] for x in range(0, len(self.constant_groups), 4)
        ]

        tables = []
        # Loop over groups of calibrations.
        for cal_group in cal_groups:
            table = [["Modules"] + cal_group]

            # Loop over calibrations and modules to form the next rows.
            for mod in modules:
                mod_consts = []

                for cname in cal_group:
                    try:
                        singleconst = self[cname, mod]
                    except KeyError:
                        # Constant is not available for this module.
                        mod_consts.append("—")
                    else:
                        # Have the creation time a reference
                        # link to the CCV on CALCAT.
                        c_time = datetime.fromisoformat(
                            singleconst.metadata("begin_validity_at")).strftime(
                                "%Y-%m-%d %H:%M")
                        try:
                            view_url = singleconst.metadata("view_url")
                            mod_consts.append((c_time, view_url))
                        except KeyError:
                            mod_consts.append(f"{c_time} ({singleconst.ccv_id})")

                table.append([str(mod)] + mod_consts)

            tables.append(table)

        return DisplayTables(tables)

    def markdown_table(self, module_naming="modnum") -> str:
        """Make a markdown table overview of the constants found.

        Columns are calibration types, rows are modules.
        If there are >4 calibrations, the table will be split up into several
        pieces with up to 4 calibrations in each.

        Args:
            module_naming (str): modnum, aggregator or qm, to change how the
                modules are labelled in the table. Defaults to modnum.
        """
        return self.summary_table(module_naming)._repr_markdown_()

    def display_markdown_table(self, module_naming="modnum"):
        """Display a table of the constants found (in a Jupyter notebook).

        Columns are calibration types, rows are modules.
        If there are >4 calibrations, the table will be split up into several
        pieces with up to 4 calibrations in each.

        Args:
            module_naming (str): modnum, aggregator or qm, to change how the
                modules are labelled in the table. Defaults to modnum.
        """
        from IPython.display import display, Markdown
        display(Markdown(self.markdown_table(module_naming=module_naming)))

    def reports_info(self):
        """Display information about the reports of found constants
        """
        by_rept_id = {}
        for cal, mmc in self.constant_groups.items():
            for mod, sc in mmc.items():
                rid = sc.metadata('report_id')
                by_rept_id.setdefault(sc.metadata('report_id'), []).append((sc, cal, mod))

        tbl = [['Report ID', 'Calibration types', 'Modules', '# constants', ]]
        for report_id, consts in by_rept_id.items():
            cals = sorted(set(t[1] for t in consts))
            mods = sorted(set(t[2] for t in consts))

            if report_id is not None:
                # This is ugly, but avoids assuming production CalCat
                ccv_url = consts[0][0].metadata('view_url')
                calcat_base_url = ccv_url.split('/calibration_constant_versions/')[0]
                report_details = str(report_id), f"{calcat_base_url}/reports/{report_id}"
            else:
                report_details = "(No report)"

            mods_summary = [f"{g[0]}–{g[-1]}" if len(g) > 1 else g[0]
                            for g in _summarise_mod_names(mods)]

            tbl.append([report_details, ', '.join(cals), ', '.join(mods_summary), str(len(consts))])

        return DisplayTables([tbl])


def tex_escape(text):
    """
    Escape latex special characters found in the text

    :param text: a plain text message
    :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
        '—': r'\textemdash',
    }

    key_list = sorted(conv.keys(), key=lambda item: - len(item))
    regex = re.compile('|'.join(re.escape(str(key)) for key in key_list))
    return regex.sub(lambda match: conv[match.group()], text)


class DisplayTables:
    def __init__(self, tables):
        # list (tables) of lists (rows) of lists (cells). A cell may be str,
        # or a (text, url) tuple to make a link.
        self.tables = tables

    @staticmethod
    def _build_table(table, fmt_link, escape=lambda s: s):
        res = []
        for row in table:
            prepd_row = []
            for cell in row:
                if isinstance(cell, tuple):
                    text, url = cell
                else:
                    text = cell
                    url = None

                if url is None:
                    prepd_row.append(escape(text))
                else:
                    prepd_row.append(fmt_link(escape(text), url))
            res.append(prepd_row)
        return res

    def _repr_markdown_(self):
        from tabulate import tabulate

        def fmt_link(text, url):
            return f"[{text}]({url})"

        prepd_tables = [self._build_table(table, fmt_link) for table in self.tables]

        return '\n\n'.join(
            tabulate(t, tablefmt="pipe", headers="firstrow")
            for t in prepd_tables
        )

    def _repr_latex_(self):
        from tabulate import tabulate

        def fmt_link(text, url):
            return r'\href{%s}{%s}' % (url, text)

        prepd_tables = [self._build_table(table, fmt_link, escape=tex_escape) for table in self.tables]

        return '\n\n'.join(
            tabulate(t, tablefmt="latex_raw", headers="firstrow")
            for t in prepd_tables
        )


def lpd_dark_consts_with_fallback(
    condition: "LPDConditions",
    detector_name,
    event_at=None,
    preference_time=timedelta(days=7),
    **kwargs
):
    """Look up LPD dark constants with fallback to constants for all memory cells

    The parameters are mostly the same as for CalibrationData.from_condition().
    Constants with the matching memory cell order will be used if they're closer
    in time than the fallback, or up to *preference_time* (default 5 days)
    further.
    """
    if event_at is None:
        event_at = datetime.now(timezone.utc)
    elif isinstance(event_at, str):
        event_at = datetime.fromisoformat(event_at)

    cd_preferred = CalibrationData.from_condition(
        condition,
        detector_name,
        calibrations=["Offset", "Noise", "BadPixelsDark"],
        event_at=event_at,
        **kwargs
    )

    fallback_mem_cell_order = ",".join([str(i) for i in range(510)]) + ","
    if condition.memory_cell_order == fallback_mem_cell_order:
        return cd_preferred  # Fallback would be the same

    cd_fallback = CalibrationData.from_condition(
        replace(condition, memory_cell_order=fallback_mem_cell_order),
        detector_name,
        calibrations=["Offset", "Noise", "BadPixelsDark"],
        event_at=event_at,
        **kwargs
    )

    def time_distance(cd, aggregator):
        bva = cd['Offset', aggregator].metadata("begin_validity_at")
        return abs(datetime.fromisoformat(bva) - event_at)

    # Select which modules to use from the fallback. In most cases, this will
    # be either none or all, as we'll have the same timestamps for all modules,
    # but it may vary if a module is replaced. In theory the timestamps can also
    # be different for each calibration type, but that's unlikely, so we just
    # look at the offset timestamps.
    aggrs_fallback = []
    for da in (set(cd_preferred.aggregator_names) | set(cd_fallback.aggregator_names)):
        if da not in cd_fallback.get('Offset', {}):
            continue
        elif da not in cd_preferred.get('Offset', {}):
            aggrs_fallback.append(da)  # Only in fallback
        else:
            closer_by = time_distance(cd_preferred, da) - time_distance(cd_fallback, da)
            if closer_by > preference_time:
                aggrs_fallback.append(da)

    return cd_preferred.merge(
        cd_fallback.select_modules(aggregator_names=aggrs_fallback)
    )

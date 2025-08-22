
import json
import re
from collections.abc import Mapping, Iterable
from dataclasses import (
    dataclass, field, fields, replace, is_dataclass, MISSING)
from datetime import date, datetime, time, timezone
from enum import IntFlag
from fnmatch import fnmatch
from functools import lru_cache
from operator import index
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin
from warnings import warn

import numpy as np
import h5py
import pasha as psh
import requests
from extra_data.read_machinery import find_proposal
from oauth2_xfel_client import Oauth2ClientBackend

from extra_data import PropertyNameError

from .utils.misc import _isinstance_no_import


__all__ = [
    "CalCatAPIError",
    "CalCatAPIClient",
    "SingleConstant",
    "MultiModuleConstant",
    "CalibrationData",
    "AGIPDConditions",
    "LPDConditions",
    "DSSCConditions",
    "JUNGFRAUConditions",
    "ShimadzuHPVX2Conditions",
    "DetectorData",
    "DetectorModule"
]

# Default address to connect to, only available internally
CALCAT_PROXY_URL = "http://exflcalproxy.desy.de:8080/"


def any_is_none(*values):
    for value in values:
        if value is None:
            return True

    return False


class ModuleNameError(KeyError):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"No module named {self.name!r}"


class NoDimensionLabelsError(ValueError):
    pass


class AutoConditionsError(ValueError):
    """Used when detector conditions could not be inferred from data."""

    def __init__(self, missing, sources):
        self.missing = missing
        self.sources = sources

    def __str__(self):
        msg = 'required parameters could not be inferred from data: ' + \
            ', '.join(sorted(self.missing))

        if self.sources:
            msg += '\n\navailable sources to infer parameters:\n'

            for name, value in self.sources.items():
                if isinstance(value, Iterable) and not isinstance(value, str):
                    if len(value) > 1:
                        value = '{}, ... [{} more]'.format(
                            sorted(value)[0], len(value) - 1)

                    elif len(value) == 1:
                        value = str(next(iter(value)))

                    elif not value:
                        value = None

                msg += f'- {name}: {value}\n'

        return msg


class CalCatAPIError(requests.HTTPError):
    """Used when the response includes error details as JSON"""
    @property
    def status_code(self):
        return self.response.status_code


class CalCatAPIClient:
    def __init__(self, base_api_url, oauth_client=None, user_email=""):
        if oauth_client is not None:
            self.oauth_client = oauth_client
            self.session = self.oauth_client.session
        else:
            # Oauth disabled - used with base_api_url pointing to an
            # xfel-oauth-proxy instance
            self.oauth_client = None
            self.session = requests.Session()

        self.user_email = user_email
        # Ensure the base URL has a trailing slash
        self.base_api_url = base_api_url.rstrip("/") + "/"

    def __repr__(self):
        auth = " (with Oauth)" if self.oauth_client else ""
        return f"<CalCatAPIClient for {self.base_api_url}{auth}>"

    def default_headers(self):
        from . import __version__
        return {
            "content-type": "application/json",
            "Accept": "application/json; version=2",
            "X-User-Email": self.user_email,
            "User-Agent": f"EXtra/{__version__}",
        }

    @classmethod
    def format_time(cls, dt):
        """Parse different ways to specify time to CalCat."""

        if isinstance(dt, datetime):
            return dt.astimezone(timezone.utc).isoformat()
        elif isinstance(dt, date):
            return cls.format_time(datetime.combine(dt, time()))
        elif _isinstance_no_import(dt, 'extra_data', 'DataCollection'):
            return cls.format_time(dt[0].train_timestamps(pydatetime=True)[0])
        elif dt is None:
            return ""  # Not specified - for searches, this usually means now
        elif not isinstance(dt, str):
            raise TypeError(
                f"Timestamp parameter ({dt!r}) must be a string, datetime or "
                f"date object"
            )

        return dt

    def request(self, method, relative_url, params=None, headers=None, **kwargs):
        """Make a GET request, return the HTTP response object"""
        # Base URL may include e.g. '/api/'. This is a prefix for all URLs;
        # even if they look like an absolute path.
        url = urljoin(self.base_api_url, relative_url.lstrip("/"))
        _headers = self.default_headers()
        if headers:
            _headers.update(headers)
        return self.session.request(
            method, url, params=params, headers=_headers, **kwargs
        )

    @staticmethod
    def _parse_response(resp: requests.Response):
        if resp.status_code >= 400:
            try:
                d = json.loads(resp.content.decode("utf-8"))
            except Exception:
                resp.raise_for_status()
            else:
                raise CalCatAPIError(
                    f"Error {resp.status_code} from API: "
                    f"{d.get('info', 'missing details')}",
                    response=resp
                )

        if resp.content == b"":
            return None
        else:
            return json.loads(resp.content.decode("utf-8"))

    def get(self, relative_url, params=None, **kwargs):
        """Make a GET request, return response content from JSON"""
        resp = self.request('GET', relative_url, params, **kwargs)
        return self._parse_response(resp)

    _pagination_headers = (
        "X-Total-Pages",
        "X-Count-Per-Page",
        "X-Current-Page",
        "X-Total-Count",
    )

    def get_paged(self, relative_url, params=None, **kwargs):
        """Make a GET request, return response content & pagination info"""
        resp = self.request('GET', relative_url, params, **kwargs)
        content = self._parse_response(resp)
        pagination_info = {
            k[2:].lower().replace("-", "_"): int(resp.headers[k])
            for k in self._pagination_headers
            if k in resp.headers
        }
        return content, pagination_info

    def post(self, relative_url, json, **kwargs):
        """Make a POST request, return response content from JSON"""
        resp = self.request('POST', relative_url, json=json, **kwargs)
        return self._parse_response(resp)

    # ------------------
    # Cached wrappers for simple ID lookups of fixed-ish info
    #
    # N.B. lru_cache behaves oddly with instance methods (it's a global cache,
    # with the instance as part of the key), but in this case it should be OK.
    @lru_cache()
    def calibration_by_id(self, cal_id):
        return self.get(f"calibrations/{cal_id}")

    @lru_cache()
    def detector_by_id(self, det_id):
        return self.get(f"detectors/{det_id}")

    # --------------------
    # Shortcuts to find 1 of something by an ID-like field (e.g. name) other
    # than CalCat's own integer IDs. Error on no match or >1 matches.
    @lru_cache
    def _get_by_name(self, endpoint, name, name_key="name"):
        res = self.get(endpoint, {name_key: name})
        if not res:
            raise KeyError(f"No {endpoint[:-1]} with name {name}")
        elif len(res) > 1:
            raise ValueError(f"Multiple {endpoint} found with name {name}")
        return res[0]

    def detector_by_identifier(self, identifier):
        return self._get_by_name(
            "detectors", identifier, name_key="identifier")

    def instrument_by_name(self, name):
        return self._get_by_name("instruments", name, name_key='identifier')

    def calibration_by_name(self, name):
        return self._get_by_name("calibrations", name)

    def parameter_by_name(self, name):
        return self._get_by_name("parameters", name)

    def detector_type_by_name(self, name):
        return self._get_by_name("detector_types", name)

    def pdu_by_name(self, name):
        return self._get_by_name(
            "physical_detector_units", name, name_key="physical_name")

    def pdus_by_detector(self, det_id, pdu_snapshot_at):
        return self.get(f'physical_detector_units/get_all_by_detector',
                        {'detector_id': det_id,
                         'event_at': self.format_time(pdu_snapshot_at)})


global_client = None


def get_client():
    """Get the global CalCat API client.

    The default assumes we're running in the DESY network; this is used unless
    `setup_client()` has been called to specify otherwise.
    """
    global global_client
    if global_client is None:
        setup_client(CALCAT_PROXY_URL, None, None, None)
    return global_client


def setup_client(
        base_url,
        client_id,
        client_secret,
        user_email,
        scope="",
        session_token=None,
        oauth_retries=3,
        oauth_timeout=12,
        ssl_verify=True,
):
    """Configure the global CalCat API client."""
    global global_client
    if client_id is not None:
        oauth_client = Oauth2ClientBackend(
            client_id=client_id,
            client_secret=client_secret,
            scope=scope,
            token_url=f"{base_url}/oauth/token",
            session_token=session_token,
            max_retries=oauth_retries,
            timeout=oauth_timeout,
            ssl_verify=ssl_verify,
        )
    else:
        oauth_client = None
    global_client = CalCatAPIClient(
        f"{base_url}/api/",
        oauth_client=oauth_client,
        user_email=user_email,
    )

    # Check we can connect to exflcalproxy
    if oauth_client is None and base_url == CALCAT_PROXY_URL:
        try:
            # timeout=(connect_timeout, read_timeout)
            global_client.request("GET", "me", timeout=(1, 5))
        except requests.ConnectionError as e:
            raise RuntimeError(
                "Could not connect to calibration catalog proxy. This proxy allows "
                "unauthenticated access inside the XFEL/DESY network. To look up "
                "calibration constants from outside, you will need to create an Oauth "
                "client ID & secret in the CalCat web interface. You will still not "
                "be able to load constants without the constant store folder."
            ) from e


_default_caldb_root = None

def set_default_caldb_root(p: Path):
    """Override the default root directory for constants in CalCat"""
    global _default_caldb_root
    _default_caldb_root = p

def get_default_caldb_root():
    """Get the root directory for constants in CalCat.

    The default location is different on Maxwell & ONC; this checks which one
    exists. Calling ``set_default_caldb_root()`` overrides this.
    """
    global _default_caldb_root
    if _default_caldb_root is None:
        onc_path = Path("/common/cal/caldb_store")
        maxwell_path = Path("/gpfs/exfel/d/cal/caldb_store")
        if onc_path.is_dir():
            _default_caldb_root = onc_path
        elif maxwell_path.is_dir():
            _default_caldb_root = maxwell_path
        else:
            raise RuntimeError(
                f"Neither {onc_path} nor {maxwell_path} was found. If the caldb_store "
                "directory is at another location, pass its path as caldb_root."
            )

    return _default_caldb_root


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
        self.condition = condition

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
                'source_name_pattern': None, 'number_of_modules': None,
                'first_module_index': None
            }

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
            modules=None,
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
        det = DetectorData(detector_name)

        try:
            cond_cls = detector_cond_cls[det.detector_type]
        except KeyError:
            raise NotImplementedError(det.detector_type)

        return cls.from_condition(
            cond_cls.from_data(data, detector_name, modules=modules,
                               client=client, **kwargs),
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
            module_number_at_ccv_begin_at=pdu.module_number,
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


class ConditionsBase:
    calibration_types = {}  # For subclasses: {calibration: [parameter names]}

    @classmethod
    def from_data(cls, params, **sources):
        if is_dataclass(cls):
            # Try to give a nicer error message if the conditions type
            # is a dataclass.
            required_fields = {f.name for f in fields(cls) if
                               bool(f.init and f.default is MISSING and
                                    f.default_factory is MISSING)}

            if (missing := required_fields - params.keys()):
                raise AutoConditionsError(missing, sources)

        return cls(**params)

    def make_dict(self, parameters) -> dict:
        d = dict()

        for db_name in parameters:
            value = getattr(self, db_name.lower().replace(" ", "_"))
            if isinstance(value, str):
                d[db_name] = value
            elif value is not None:
                d[db_name] = float(value)

        return d

    def _repr_markdown_(self):
        attr_names = [f.name for f in fields(self)]
        items = []
        for n in attr_names:
            if (value := getattr(self, n)) is not None:
                items.append(f"- {n.replace('_', ' ').capitalize()}: {value}")
        return '\n'.join(items)

    @staticmethod
    def _find_detector_modules(data, det):
        assert det['first_module_index'] is not None and \
            det['number_of_modules'] is not None, \
            'incomplete detector entry in CalCat'

        # Find module numbers present in data.
        return {
            modno for modno in range(
                det['first_module_index'],
                det['first_module_index'] + det['number_of_modules'])
            if det['source_name_pattern'].format(modno=modno)
                in data.instrument_sources}

    @staticmethod
    def _purge_missing_sources(data, *sources):
        result = []

        for src in sources:
            if isinstance(src, str):
                result.append(src if src in data.all_sources else None)
            else:
                result.append([s for s in src if s in data.all_sources])

        return result


@dataclass
class AGIPDConditions(ConditionsBase):
    """Conditions for AGIPD detectors"""
    sensor_bias_voltage: float
    memory_cells: int
    acquisition_rate: float
    gain_setting: Optional[int]
    gain_mode: Optional[int]
    source_energy: float = 9.2
    integration_time: int = 12
    pixels_x: int = 512
    pixels_y: int = 128

    _gain_parameters = [
        "Sensor Bias Voltage",
        "Pixels X",
        "Pixels Y",
        "Memory cells",
        "Acquisition rate",
        "Gain setting",
        "Integration time",
    ]
    _other_dark_parameters = _gain_parameters + ["Gain mode"]
    _illuminated_parameters = _gain_parameters + ["Source energy"]

    calibration_types = {
        "Offset": _other_dark_parameters,
        "Noise": _other_dark_parameters,
        "ThresholdsDark": _other_dark_parameters,
        "BadPixelsDark": _other_dark_parameters,
        "BadPixelsPC": _gain_parameters,
        "SlopesPC": _gain_parameters,
        "BadPixelsFF": _illuminated_parameters,
        "SlopesFF": _illuminated_parameters,
    }

    def make_dict(self, parameters):
        cond = super().make_dict(parameters)

        # Fix-up some database quirks.
        if int(cond.get("Gain mode", -1)) == 0:
            del cond["Gain mode"]

        if int(cond.get("Integration time", -1)) == 12:
            del cond["Integration time"]

        return cond

    @classmethod
    def from_data(cls, data, detector_name, modules=None,
                  fpga_comp=None, mpod=None, fpga_control=None, xtdf=None,
                  client=None, **params):
        # Uncaught exceptions that may be thrown in here:
        # - NoDataError

        if any_is_none(modules, fpga_comp, mpod, fpga_control, xtdf):
            detector = (client or get_client()).detector_by_identifier(
                detector_name)
            control_domain = detector['karabo_id_control']

            modules = modules or cls._find_detector_modules(data, detector)
            fpga_comp = fpga_comp or f'{control_domain}/MDL/FPGA_COMP'
            mpod = mpod or f'{control_domain[:-1]}/PSC/HV'

            if fpga_control is None:
                fpga_control = {f'{control_domain}/FPGA/M_{modno}'
                                for modno in modules}

            if xtdf is None:
                xtdf = {detector['source_name_pattern'].format(modno=modno)
                        for modno in modules}

        fpga_comp, mpod, fpga_control, xtdf = cls._purge_missing_sources(
            data, fpga_comp, mpod, fpga_control, xtdf)

        if fpga_comp is not None:
            # Prefer control data from FPGA composite device.
            sd = data[fpga_comp]

            if 'memory_cells' not in params:
                params['memory_cells'] = cls.memory_cells_from_comp(sd)

            if 'acquisition_rate' not in params:
                params['acquisition_rate'] = cls.acquisition_rate_from_comp(sd)

            if 'gain_setting' not in params:
                params['gain_setting'] = cls.gain_setting_from_comp(sd)

            if 'gain_mode' not in params:
                params['gain_mode'] = cls.gain_mode_from_comp(sd)

            if 'integration_time' not in params:
                try:
                    val = cls.integration_time_from_comp(sd)
                except PropertyNameError:
                    # More recent feature of comp device, fallback to
                    # prior defalt.
                    val = 12

                params['integration_time'] = val

        elif xtdf:
            # Fallback to estimate some parameters from XTDF data.
            if 'memory_cells' not in params:
                for src in xtdf:
                    if (val := cls.memory_cells_from_xtdf(data[src])) > 0:
                        break

                params['memory_cells'] = val

            if 'acquisition_rate' not in params:
                for src in xtdf:
                    if (val := cls.acquisition_rate_from_xtdf(data[src])) > 0:
                        break

                params['acquisition_rate'] = val

        if 'sensor_bias_voltage' not in params:
            if mpod is not None:
                # AGIPD Gen1
                params['sensor_bias_voltage'] = cls.bias_voltage_from_mpod(
                    data[mpod], modules)

            elif fpga_control:
                # AGIPD Gen2
                for src in fpga_control:
                    if (val := cls.bias_voltage_from_control(data[src])) > 0:
                        break

                params['sensor_bias_voltage'] = val

        return super().from_data(params,
                                 fpga_comp=fpga_comp, mpod=mpod,
                                 fpga_control=fpga_control, xtdf=xtdf)

    @staticmethod
    def bias_voltage_from_control(sd):
        # These device used to suffer from switching to excessive values
        # randomly, so cut off any unreasonable values.
        values = sd['highVoltage.actual'].ndarray()
        values = values[values < 1000.0]
        return int(np.median(values))

    @staticmethod
    def bias_voltage_from_mpod(sd, modules=None):
        if modules is not None:
            keys = [f'channels.U{modno}.measurementSenseVoltage'
                    for modno in modules]
        else:
            keys = sd.select_keys(
                'channels.U*.measurementSenseVoltage').keys(False)

        for key in keys:
            if (val := sd[key].as_single_value(atol=1, rtol=1)) > 0:
                return int(val)

    @staticmethod
    def memory_cells_from_comp(sd):
        return int(sd['bunchStructure.nPulses']
            .as_single_value(reduce_by='max'))

    @staticmethod
    def memory_cells_from_xtdf(sd):
        # Only look at one train?
        cell_ids = sd['image.cellId'].drop_empty_trains().ndarray().squeeze()
        options = np.array([4, 32, 64, 76, 128, 176, 202, 250, 352])
        return int(options[np.flatnonzero(options > np.max(cell_ids)).min()])

    @staticmethod
    def acquisition_rate_from_comp(sd):
        return round(float(sd['bunchStructure.repetitionRate']
            .as_single_value()), 1)

    @staticmethod
    def acquisition_rate_from_xtdf(sd):
        pulse_ids = sd['image.pulseId'].drop_empty_trains().ndarray().squeeze()
        return round(np.floor(45 / np.diff(pulse_ids).min()) / 10, 1)

    @staticmethod
    def gain_setting_from_comp(sd):
        if 'gain' in sd:
            return int(sd['gain'].as_single_value(atol=0))

        # Legacy method for older versions of composite device.

        setupr = sd['setupr'].as_single_value()
        pattern_type_idx = sd['patternTypeIndex'].as_single_value()

        if (setupr == 0 and pattern_type_idx < 4):
            return 0
        elif (setupr == 32 and pattern_type_idx == 4):
            return 0
        elif (setupr == 8 and pattern_type_idx < 4):
            return 1
        elif (setupr == 40 and pattern_type_idx == 4):
            return 1

        raise ValueError('unexpected setupr and patternTypeIndex values to '
                         'determine CDS mode')

    @staticmethod
    def gain_mode_from_comp(sd):
        return int(sd['gainModeIndex'].as_single_value(atol=0))

    @staticmethod
    def integration_time_from_comp(sd):
        return int(sd['integrationTime'].as_single_value())


@dataclass
class LPDConditions(ConditionsBase):
    sensor_bias_voltage: float = 250.0
    memory_cells: int = 512
    memory_cell_order: Optional[str] = None
    feedback_capacitor: float = 5.0
    source_energy: float = 9.3
    category: int = 0
    pixels_x: int = 256
    pixels_y: int = 256
    parallel_gain: bool = False

    _base_params = [
        "Sensor Bias Voltage",
        "Memory cells",
        "Pixels X",
        "Pixels Y",
        "Feedback capacitor",
    ]
    _dark_parameters = _base_params + [
        "Memory cell order", "Parallel gain"
    ]
    _illuminated_parameters = _base_params + ["Source Energy", "category"]

    calibration_types = {
        "Offset": _dark_parameters,
        "Noise": _dark_parameters,
        "BadPixelsDark": _dark_parameters,
        "RelativeGain": _illuminated_parameters,
        "GainAmpMap": _illuminated_parameters,
        "FFMap": _illuminated_parameters,
        "BadPixelsFF": _illuminated_parameters,
    }

    def make_dict(self, parameters):
        cond = super().make_dict(parameters)

        # Legacy value for no parallel gain not injected for backwards
        # compatibility with prior calibration data.
        if int(cond.get("Parallel gain", -1)) == 0:
            del cond["Parallel gain"]

        return cond

    @classmethod
    def from_data(cls, data, detector_name, modules=None,
                  fem_comp=None, xtdf=None,
                  validate_memory_order=False,
                  client=None, **params):
        if any_is_none(modules, fem_comp, xtdf):
            detector = (client or get_client()).detector_by_identifier(
                detector_name)

            modules = modules or cls._find_detector_modules(data, detector)
            fem_comp = fem_comp or \
                detector['karabo_id_control'] + '/COMP/FEM_MDL_COMP'

            if xtdf is None:
                xtdf = {detector['source_name_pattern'].format(modno=modno)
                        for modno in modules}

        fem_comp, xtdf = cls._purge_missing_sources(data, fem_comp, xtdf)

        if fem_comp is not None and 'parallel_gain' not in params:
            try:
                val = cls.parallel_gain_from_fem_comp(data[fem_comp])
            except PropertyNameError:
                # Added to the FEM comp with introduction of parallel
                # gain support.
                val = False

            params['parallel_gain'] = val

        if xtdf and 'memory_cell_order' not in params:
            prev_val = None  # used with validate_memory_order

            for src in xtdf:
                if (val := cls.memory_cell_order_from_xtdf(data[src])).size:
                    if validate_memory_order:
                        if prev_val is not None and (prev_val != val).any():
                            raise ValueError('inconsistent memory order '
                                             'across modules')

                        prev_val = val
                    else:
                        break

            params['memory_cell_order'] = val

        return super().from_data(params, fem_comp=fem_comp, xtdf=xtdf)

    @staticmethod
    def gain_mode_from_fem_comp(sd):
        # Not actually used in the condition at the moment.
        return int(sd.run_value('femAsicGain'))

    @staticmethod
    def parallel_gain_from_fem_comp(sd):
        return bool(sd.run_value('femAsicGainOverride'))

    @staticmethod
    def memory_cell_order_from_xtdf(sd):
        return sd['image.cellId'].drop_empty_trains()[0].ndarray().flatten()


@dataclass
class DSSCConditions(ConditionsBase):
    """Conditions for DSSC detectors"""
    sensor_bias_voltage: float
    memory_cells: int
    pulse_id_checksum: Optional[float] = None
    acquisition_rate: Optional[float] = None
    target_gain: Optional[int] = None
    encoded_gain: Optional[int] = None
    pixels_x: int = 512
    pixels_y: int = 128

    _params = [
        "Sensor Bias Voltage",
        "Memory cells",
        "Pixels X",
        "Pixels Y",
        "Pulse id checksum",
        "Acquisition rate",
        "Target gain",
        "Encoded gain",
    ]
    calibration_types = {
        "Offset": _params,
        "Noise": _params,
    }


@dataclass
class JUNGFRAUConditions(ConditionsBase):
    """Conditions for JUNGFRAU detectors"""
    sensor_bias_voltage: float
    memory_cells: int
    integration_time: float
    gain_setting: int
    gain_mode: Optional[int] = None
    exposure_timeout: int = 25
    sensor_temperature: float = 291
    pixels_x: int = 1024
    pixels_y: int = 512

    _params = [
        "Sensor Bias Voltage",
        "Memory Cells",
        "Pixels X",
        "Pixels Y",
        "Integration Time",
        "Sensor temperature",
        "Gain Setting",
        "Gain mode",
    ]
    _dark_params = _params + ["Exposure timeout"]

    calibration_types = {
        "Offset10Hz": _dark_params,
        "Noise10Hz": _dark_params,
        "BadPixelsDark10Hz": _dark_params,
        "RelativeGain10Hz": _params,
        "BadPixelsFF10Hz": _params,
    }

    # Before 2022, the settings key indicated both gain
    # mode (as in adaptive vs fixed gain) as well as gain
    # setting (as in high CDS or not). Since then, there
    # is a dedicated gainMode key and settings only
    # indicates high CDS.
    # See karaboDevices/slsDetectors@4433ae9c00edcca3309bec8b7515e0938f5f502c
    legacy_settings = {
        # old setting:  new settings, new gainMode
        'dynamicgain': ('gain0', 'dynamic'),
        'dynamichg0':  ('highgain0', 'dynamic'),
        'fixgain1': ('gain0', 'fixg1'),
        'fixgain2': ('gain0', 'fixg2'),
        'forceswitchg1': ('gain0', 'forceswitchg1'),
        'forceswitchg2': ('gain0', 'forceswitchg2'),
    }

    gain_mode_labels = {
        'dynamic': 0,
        'fixg0': 1,
        'fixg1': 2,
        'fixg2': 3,

        # forceswitchg1, forceswitchg2 may only be used for
        # darks and are not equivalent to their fixed gain
        # equivalents.
    }

    def make_dict(self, parameters):
        cond = super().make_dict(parameters)

        # Fix-up some database quirks.
        if int(cond.get("Gain mode", -1)) == 0:
            del cond["Gain mode"]

        # Fix-up some database quirks.
        if int(cond.get("Exposure timeout", -1)) == 25:
            del cond["Exposure timeout"]

        return cond

    @classmethod
    def from_data(cls, data, detector_name, modules=None,
                  control=None,
                  client=None, **params):
        if control is None:
            detector = (client or get_client()).detector_by_identifier(
                detector_name)
            control = control or '{}/DET/CONTROL'.format(detector['karabo_id_control'])

        control, = cls._purge_missing_sources(data, control)

        if control is not None:
            sd = data[control]

            if 'sensor_bias_voltage' not in params:
                params['sensor_bias_voltage'] = cls.sensor_bias_voltage_from_control(sd)

            if 'memory_cells' not in params:
                params['memory_cells'] = cls.memory_cells_from_control(sd)

            if 'integration_time' not in params:
                params['integration_time'] = cls.integration_time_from_control(sd)

            if 'exposure_timeout' not in params:
                params['exposure_timeout'] = cls.exposure_timeout_from_control(sd)

            if 'gain_setting' not in params:
                params['gain_setting'] = cls.gain_setting_from_control(sd)

            if 'gain_mode' not in params:
                params['gain_mode'] = cls.gain_mode_from_control(sd)

        return super().from_data(params, control=control)

    @staticmethod
    def sensor_bias_voltage_from_control(sd):
        for key in ['highVoltage', 'vHighVoltage']:
            if key not in sd:
                continue

            return int(sd.run_value(key)[0])

        raise PropertyNameError('highVoltage or vHighVoltage', sd.source)

    @staticmethod
    def memory_cells_from_control(sd):
        return int(sd.run_value('storageCells')) + 1

    @staticmethod
    def memory_cell_start_from_control(sd):
        # Not used in condition, but relevant for dark characterization.
        return int(sd.run_value('storageCellStart'))

    @staticmethod
    def integration_time_from_control(sd):
        return 1e6 * float(sd.run_value('exposureTime'))

    @staticmethod
    def exposure_timeout_from_control(sd):
        return int(sd.run_value('exposureTimeout'))

    @classmethod
    def gain_setting_from_control(cls, sd, raw=False):
        val = sd.run_value('settings')

        if 'gainMode' not in sd:
            # Convert from legacy value.
            val = cls.legacy_settings[val][0]

        if raw:
            return val

        return int(val == 'highgain0')

    @classmethod
    def gain_mode_from_control(cls, sd, raw=False):
        try:
            val = sd.run_value('gainMode')
        except PropertyNameError:
            val = cls.legacy_settings[sd.run_value('settings')][1]

        if raw:
            return val

        try:
            return cls.gain_mode_labels[val]
        except KeyError:
            raise ValueError(f'invalid gain mode {val!s} encountered') from None


@dataclass
class ShimadzuHPVX2Conditions(ConditionsBase):
    burst_frame_count: float

    calibration_types = {
        'Offset': ['Burst Frame Count'],
        'DynamicFF': ['Burst Frame Count'],
    }


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
    virtual_device_name: str
    module_index: int
    module_number: int | None
    detector_type: str
    legacy_uuid: int | None  # Deprecated, do not use

    def __post_init__(self):
        if self.module_number is None:
            # Try to fill in module number if missing.
            self.module_number = int(re.findall(r"\d+", self.aggregator)[-1])

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

    Attributes:
        id (int): Detector numerical ID.
        identifier (str): Detector identifier.
        number_of_modules (int): Number of modules for the full detector,
            may be more than currently installed or selected.
        pdu_snapshot_at (str): ISO format date the mapping is taken from.
    """

    def __init__(self, detector_row, module_rows_or_pdus, pdu_snapshot_at=None):
        # Result rows as returned by CalCat.

        self.id = detector_row['id']
        self.identifier = detector_row['identifier']
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
                    item['uuid'])
            else:
                item.module_index = i

            self.pdus.append(item)

        self.pdu_snapshot_at = pdu_snapshot_at or datetime.now().isoformat()

    @classmethod
    def _from_detector(cls, detector_row, pdu_snapshot_at, client):
        pdu_snapshot_at = client.format_time(pdu_snapshot_at)

        try:
            module_rows = client.get(
                'physical_detector_units/get_all_by_detector',
                {'detector_id': detector_row['id'],
                 'pdu_snapshot_at': pdu_snapshot_at})
        except CalCatAPIError as e:
            if e.status_code == 404:
                module_rows = []
            else:
                raise e

        return cls(detector_row, module_rows, pdu_snapshot_at)

    @classmethod
    def from_id(cls, detector_id, pdu_snapshot_at=None, client=None):
        """Look up a detector and its modules by CalCat ID.

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
        instrument_id = client.instrument_by_name(instrument)['id']

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

    @classmethod
    def list_by_instrument(cls, instrument, client=None):
        """List all detectors by instrument."""

        client = client or get_client()
        instrument_id = client.instrument_by_name(instrument)['id']

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
    def source_names(self) -> list[str]:
        """Source names of currently mapped PDUs."""
        return [self.source_name_pattern.format(
            modno=pdu.module_number or i + self.first_module_index
        ) for i, pdu in enumerate(self.pdus)]

    @property
    def first_module_index(self) -> int:
        """Module index of the first module."""
        assert self._first_module_index is not None, \
            'incomplete detector entry in CalCat'
        return self._first_module_index

    @property
    def pdu_detector_types(self) -> set[str]:
        """Detector types of currently installed PDUs."""
        return {pdu.detector_type for pdu in self.pdus}

    @property
    def detector_type(self) -> str:
        """Detector type of all PDUs if unique."""
        pdu_types = self.pdu_detector_types

        if len(pdu_types) > 1:
            raise ValueError('more than one type of PDU: ' +
                             ', '.join(pdu_types))
        elif len(pdu_types) == 0:
            raise ValueError('no mapped PDUs')

        return pdu_types.pop()


detector_cond_cls = {
    'AGIPD-Type': AGIPDConditions,
    'LPD-Type': LPDConditions,
    'jungfrau-Type': JUNGFRAUConditions
}


class BadPixels(IntFlag):
    """Bad pixel reasons, as used in masks in corrected detector data"""
    OFFSET_OUT_OF_THRESHOLD  = 1 << 0
    NOISE_OUT_OF_THRESHOLD   = 1 << 1
    OFFSET_NOISE_EVAL_ERROR  = 1 << 2
    NO_DARK_DATA             = 1 << 3
    CI_GAIN_OF_OF_THRESHOLD  = 1 << 4
    CI_LINEAR_DEVIATION      = 1 << 5
    CI_EVAL_ERROR            = 1 << 6
    FF_GAIN_EVAL_ERROR       = 1 << 7
    FF_GAIN_DEVIATION        = 1 << 8
    FF_NO_ENTRIES            = 1 << 9
    CI2_EVAL_ERROR           = 1 << 10
    VALUE_IS_NAN             = 1 << 11
    VALUE_OUT_OF_RANGE       = 1 << 12
    GAIN_THRESHOLDING_ERROR  = 1 << 13
    DATA_STD_IS_ZERO         = 1 << 14
    ASIC_STD_BELOW_NOISE     = 1 << 15
    INTERPOLATED             = 1 << 16
    NOISY_ADC                = 1 << 17
    OVERSCAN                 = 1 << 18
    NON_SENSITIVE            = 1 << 19
    NON_LIN_RESPONSE_REGION  = 1 << 20
    WRONG_GAIN_VALUE         = 1 << 21
    NON_STANDARD_SIZE        = 1 << 22


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

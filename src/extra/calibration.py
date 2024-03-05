import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urljoin
from warnings import warn

import h5py
import pasha as psh
import requests
from oauth2_xfel_client import Oauth2ClientBackend

__all__ = [
    "CalCatAPIError",
    "CalCatAPIClient",
    "SingleConstant",
    "MultiModuleConstant",
    "CalibrationData",
    "AGIPDConditions",
    "LPDConditions",
    "DSSCConditions",
]

# Default address to connect to, only available internally
CALCAT_PROXY_URL = "http://exflcalproxy.desy.de:8080/"


class ModuleNameError(KeyError):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"No module named {self.name!r}"


class CalCatAPIError(requests.HTTPError):
    """Used when the response includes error details as JSON"""


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
        elif not isinstance(dt, str):
            raise TypeError(
                f"Timestamp parameter ({dt!r}) must be a string, datetime or "
                f"date object"
            )

        return dt

    def get_request(self, relative_url, params=None, headers=None, **kwargs):
        """Make a GET request, return the HTTP response object"""
        # Base URL may include e.g. '/api/'. This is a prefix for all URLs;
        # even if they look like an absolute path.
        url = urljoin(self.base_api_url, relative_url.lstrip("/"))
        _headers = self.default_headers()
        if headers:
            _headers.update(headers)
        return self.session.get(url, params=params, headers=_headers, **kwargs)

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
                    f"{d.get('info', 'missing details')}"
                )

        if resp.content == b"":
            return None
        else:
            return json.loads(resp.content.decode("utf-8"))

    def get(self, relative_url, params=None, **kwargs):
        """Make a GET request, return response content from JSON"""
        resp = self.get_request(relative_url, params, **kwargs)
        return self._parse_response(resp)

    _pagination_headers = (
        "X-Total-Pages",
        "X-Count-Per-Page",
        "X-Current-Page",
        "X-Total-Count",
    )

    def get_paged(self, relative_url, params=None, **kwargs):
        """Make a GET request, return response content & pagination info"""
        resp = self.get_request(relative_url, params, **kwargs)
        content = self._parse_response(resp)
        pagination_info = {
            k[2:].lower().replace("-", "_"): int(resp.headers[k])
            for k in self._pagination_headers
            if k in resp.headers
        }
        return content, pagination_info

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
    @lru_cache()
    def detector_by_identifier(self, identifier):
        # The "identifier", "name" & "karabo_name" fields seem to have the same names
        res = self.get("detectors", {"identifier": identifier})
        if not res:
            raise KeyError(f"No detector with identifier {identifier}")
        elif len(res) > 1:
            raise ValueError(f"Multiple detectors found with identifier {identifier}")
        return res[0]

    @lru_cache()
    def calibration_by_name(self, name):
        res = self.get("calibrations", {"name": name})
        if not res:
            raise KeyError(f"No calibration with name {name}")
        elif len(res) > 1:
            raise ValueError(f"Multiple calibrations found with name {name}")
        return res[0]


global_client = None


def get_client():
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
            global_client.get_request("me", timeout=(1, 5))
        except requests.ConnectionError as e:
            raise RuntimeError(
                "Could not connect to calibration catalog proxy. This proxy allows "
                "unauthenticated access inside the XFEL/DESY network. To look up "
                "calibration constants from outside, you will need to create an Oauth "
                "client ID & secret in the CalCat web interface. You will still not "
                "be able to load constants without the constant store folder."
            ) from e


_default_caldb_root = None


def _get_default_caldb_root():
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

    def dataset_obj(self, caldb_root=None) -> h5py.Dataset:
        if caldb_root is not None:
            caldb_root = Path(caldb_root)
        else:
            caldb_root = _get_default_caldb_root()

        f = h5py.File(caldb_root / self.path, "r")
        return f[self.dataset]["data"]

    def ndarray(self, caldb_root=None):
        return self.dataset_obj(caldb_root)[:]

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
    """A group of similar constants for several modules of one detector"""

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

        for m in self.module_details:
            names = (m["module_number"], m["virtual_device_name"], m["physical_name"])
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
        return sorted(self.constants)

    @property
    def module_nums(self):
        return [
            m["module_number"]
            for m in self.module_details
            if m["karabo_da"] in self.constants
        ]

    @property
    def qm_names(self):
        return [
            m["virtual_device_name"]
            for m in self.module_details
            if m["karabo_da"] in self.constants
        ]

    @property
    def pdu_names(self):
        return [
            m["physical_name"]
            for m in self.module_details
            if m["karabo_da"] in self.constants
        ]

    def ndarray(self, caldb_root=None, *, parallel=0):
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

    def xarray(self, module_naming="modnum", caldb_root=None, *, parallel=0):
        import xarray

        if module_naming == "aggregator":
            modules = self.aggregator_names
        elif module_naming == "modnum":
            modules = self.module_nums
        elif module_naming == "qm":
            modules = self.qm_names
        else:
            raise ValueError(
                f"{module_naming=} (must be 'aggregator', 'modnum' or 'qm'"
            )

        ndarr = self.ndarray(caldb_root, parallel=parallel)

        # Dimension labels
        dims = ["module"] + ["dim_%d" % i for i in range(ndarr.ndim - 1)]
        coords = {"module": modules}
        name = self.calibration_name

        return xarray.DataArray(ndarr, dims=dims, coords=coords, name=name)


class CalibrationData(Mapping):
    """Collected constants for a given detector"""

    def __init__(self, constant_groups, module_details, detector_name):
        # {calibration: {karabo_da: SingleConstant}}
        self.constant_groups = constant_groups
        self.module_details = module_details
        self.detector_name = detector_name

    @staticmethod
    def _format_cond(condition):
        """Encode operating condition to CalCat API format.

        Args:
            condition (dict): Mapping of parameter DB name to value

        Returns:
            (dict) Operating condition for use in CalCat API.
        """

        return {
            "parameters_conditions_attributes": [
                {"parameter_name": k, "value": str(v)} for k, v in condition.items()
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
    ):
        if calibrations is None:
            calibrations = set(condition.calibration_types)
        if pdu_snapshot_at is None:
            pdu_snapshot_at = event_at

        cal_types_by_params_used = {}
        for cal_type, params in condition.calibration_types.items():
            if cal_type in calibrations:
                cal_types_by_params_used.setdefault(tuple(params), []).append(cal_type)

        client = client or get_client()

        detector_id = client.detector_by_identifier(detector_name)["id"]
        pdus = client.get(
            "physical_detector_units/get_all_by_detector",
            {
                "detector_id": detector_id,
                "pdu_snapshot_at": client.format_time(pdu_snapshot_at),
            },
        )
        module_details = sorted(pdus, key=lambda d: d["karabo_da"])
        for mod in module_details:
            if mod.get("module_number") is None:
                mod["module_number"] = int(re.findall(r"\d+", mod["karabo_da"])[-1])

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
                },
                data=json.dumps(cls._format_cond(condition_dict)),
            )

            for ccv in query_res:
                aggr = ccv["physical_detector_unit"]["karabo_da"]
                cal_type = cal_id_map[ccv["calibration_constant"]["calibration_id"]]

                const_group = constant_groups.setdefault(cal_type, {})
                const_group[aggr] = SingleConstant.from_response(ccv)

        return cls(constant_groups, module_details, detector_name)

    @classmethod
    def from_report(
            cls,
            report_id_or_path: Union[int, str],
            client=None,
    ):
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
        det_name = client.detector_by_id(det_ids.pop())["identifier"]

        module_details = sorted(pdus.values(), key=lambda d: d["karabo_da"])
        return cls(constant_groups, module_details, det_name)

    def __getitem__(self, key) -> MultiModuleConstant:
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
        return [m["module_number"] for m in self.module_details]

    @property
    def aggregator_names(self):
        return [m["karabo_da"] for m in self.module_details]

    @property
    def qm_names(self):
        return [m["virtual_device_name"] for m in self.module_details]

    @property
    def pdu_names(self):
        return [m["physical_name"] for m in self.module_details]

    def require_calibrations(self, calibrations):
        """Drop any modules missing the specified constant types"""
        mods = set(self.aggregator_names)
        for cal_type in calibrations:
            mods.intersection_update(self[cal_type].constants)
        return self.select_modules(aggregator_names=mods)

    def select_modules(
            self, module_nums=None, *, aggregator_names=None, qm_names=None
    ) -> "CalibrationData":
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
        return type(self)(constant_groups, module_details, self.detector_name)

    def select_calibrations(self, calibrations) -> "CalibrationData":
        const_groups = {c: self.constant_groups[c] for c in calibrations}
        return type(self)(const_groups, self.module_details, self.detector_name)

    def merge(self, *others: "CalibrationData") -> "CalibrationData":
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

        return type(self)(constant_groups, module_details, det_name)


class ConditionsBase:
    calibration_types = {}  # For subclasses: {calibration: [parameter names]}

    def make_dict(self, parameters) -> dict:
        d = dict()

        for db_name in parameters:
            value = getattr(self, db_name.lower().replace(" ", "_"))
            if value is not None:
                d[db_name] = value

        return d


@dataclass
class AGIPDConditions(ConditionsBase):
    sensor_bias_voltage: float
    memory_cells: int
    acquisition_rate: float
    gain_setting: Optional[int]
    gain_mode: Optional[int]
    source_energy: float
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


@dataclass
class LPDConditions(ConditionsBase):
    sensor_bias_voltage: float
    memory_cells: int
    memory_cell_order: Optional[str] = None
    feedback_capacitor: float = 5.0
    source_energy: float = 9.2
    category: int = 1
    pixels_x: int = 256
    pixels_y: int = 256

    _base_params = [
        "Sensor Bias Voltage",
        "Memory cells",
        "Pixels X",
        "Pixels Y",
        "Feedback capacitor",
    ]
    _dark_parameters = _base_params + [
        "Memory cell order",
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


@dataclass
class DSSCConditions(ConditionsBase):
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
    sensor_bias_voltage: float
    memory_cells: int
    integration_time: float
    gain_setting: int
    gain_mode: Optional[int] = None
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
    calibration_types = {
        "Offset10Hz": _params,
        "Noise10Hz": _params,
        "BadPixelsDark10Hz": _params,
        "RelativeGain10Hz": _params,
        "BadPixelsFF10Hz": _params,
    }

    def make_dict(self, parameters):
        cond = super().make_dict(parameters)

        # Fix-up some database quirks.
        if int(cond.get("Gain mode", -1)) == 0:
            del cond["Gain mode"]

        return cond

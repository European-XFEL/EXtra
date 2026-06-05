
import json
from datetime import date, datetime, time, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin

import requests
from oauth2_xfel_client import Oauth2ClientBackend

from ..utils.misc import _isinstance_no_import


# Default address to connect to, only available internally
CALCAT_PROXY_URL = "http://exflcalproxy.desy.de:8080/"

global_client = None

_default_caldb_root = None


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
        from .. import __version__
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

    def instrument_by_identifier(self, name):
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


from extra._import_proxies import generate_getattr
__getattr__ = generate_getattr(__name__, 'extra_data')

from extra_data.components import *

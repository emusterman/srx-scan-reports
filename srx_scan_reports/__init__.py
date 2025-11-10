# Module for generating a scan report at the SRX beamline
# =====================================================================
#
# This Python module is used to generate a scan report of all scans
# collected during at beamtime at the 5-ID (SRX) beamline at the NSLS-II.
#
# No other documentation is currently avialable

__author__ = ['Evan J. Musterman']
__contact__ = 'emusterma@bnl.gov'
__license__ = 'N/A'
__copyright__ = 'N/A'
__status__ = 'testing'
__date__ = "11/10/2025" # MM/DD/YYYY


# This version is not currently published!
__version__ = '0.1.0'


submodules = [

]

# This is required for wildcard (*) imports
__all__ = submodules + [
    'generate_scan_reports'
]


def __dir__():
    return __all__


# Universal only load once
from tiled.client import from_profile

# Access data
c = from_profile('srx')


# Bring class objects one level up for convenience
from .generate_scan_report import generate_scan_report
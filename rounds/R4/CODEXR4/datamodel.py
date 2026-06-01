# Local shim: re-exports IMC's datamodel from prosperity2bt so that
# `from datamodel import ...` works identically in both local backtesting
# and on the IMC platform.
from prosperity2bt.datamodel import *  # noqa: F401, F403

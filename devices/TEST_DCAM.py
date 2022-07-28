import ctypes
import ctypes.util
import numpy as np
import time
# import Hamamatsu_ScopeFoundry.CameraHardware
from hardware.CameraHardware import *
from numpy import log2
print(ctypes.windll.captureservice)
dcam = ctypes.windll.dcamapi
print(dcam)

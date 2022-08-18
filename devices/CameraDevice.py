"""
Intro:  Device driver of Hamamatsu camera.
        Camera model: Orca Fusion
Author: Hai Gong
Email:  h.gong@imperial.ac.uk
Time:   Oct 2020
Address:Imperial College London

Modified from:
    Written by Michele Castriotta, Alessandro Zecchi, Andrea Bassi (Polimi).
    Code for creating the app class of ScopeFoundry for the Orca Flash 4V3
    11/18
"""

import ctypes
import ctypes.util
import numpy as np
import time
from hardware.CameraHardware import *
from numpy import log2

# Hamamatsu constants.

# DCAM4 API.
DCAMERR_ERROR = 0
DCAMERR_NOERROR = 1
# DCAMERR_INVALIDPARAM = int("0x80000808", 0)
DCAMERR_INVALIDPARAM = ctypes.c_int32(0x80000808).value #done this way because ctypes convert properly the hexadecimal (I think it's a problem of two complement convention)
DCAMERR_BUSY = ctypes.c_int32(0x80000101).value

err_dict = {ctypes.c_int32(0x80000808).value: "DCAMERR_INVALIDPARAM",
            ctypes.c_int32(0x80000101).value: "DCAMERR_BUSY",
            ctypes.c_int32(0x80000103).value: "DCAMERR_NOTREADY",
            ctypes.c_int32(0x80000104).value: "DCAMERR_NOTSTABLE",
            ctypes.c_int32(0x80000105).value: "DCAMERR_UNSTABLE",
            ctypes.c_int32(0x80000107).value: "DCAMERR_NOTBUSY",
            ctypes.c_int32(0x80000110).value: "DCAMERR_EXCLUDED",
            ctypes.c_int32(0x80000302).value: "DCAMERR_COOLINGTROUBLE",
            ctypes.c_int32(0x80000303).value: "DCAMERR_NOTRIGGER",
            ctypes.c_int32(0x80000304).value: "DCAMERR_TEMPERATURE_TROUBLE",
            ctypes.c_int32(0x80000305).value: "DCAMERR_TOOFREQUENTTRIGGER",
            ctypes.c_int32(0x80000102).value: "DCAMERR_ABORT",
            ctypes.c_int32(0x80000106).value: "DCAMERR_TIMEOUT",
            ctypes.c_int32(0x80000301).value: "DCAMERR_LOSTFRAME",
            ctypes.c_int32(0x80000f06).value: "DCAMERR_MISSINGFRAME_TROUBLE",
            ctypes.c_int32(0x80000822).value: "DCAMERR_OUTOFRANGE",
            ctypes.c_int32(0x80000827).value: "DCAMERR_WRONGHANDSHAKE",
            ctypes.c_int32(0x83001002).value: "DCAMERR_FAILREADCAMERA",
            ctypes.c_int32(0x83001003).value: "DCAMERR_FAILWRITECAMERA",
            ctypes.c_int32(0x80000828).value: "DCAMERR_NOPROPERTY",
            ctypes.c_int32(0x80000821).value: "DCAMERR_INVALIDVALUE",
            ctypes.c_int32(0x80000833).value: "DCAMERR_INVALIDFRAMEINDEX",
            ctypes.c_int32(0x80000829).value: "DCAMERR_INVALIDCHANNEL",
            ctypes.c_int32(0x8000082a).value: "DCAMERR_INVALIDVIEW",
            ctypes.c_int32(0x8000082c).value: "DCAMERR_ACCESSDENY",
            ctypes.c_int32(0x8000082b).value: "DCAMERR_INVALIDSUBARRAY",
            ctypes.c_int32(0x8000082d).value: "DCAMERR_NOVALUETEXT",
            ctypes.c_int32(0x8000082e).value: "DCAMERR_WRONGPROPERTYVALUE",
            ctypes.c_int32(0x80000830).value: "DCAMERR_DISHARMONY",
            ctypes.c_int32(0x80000832).value: "DCAMERR_FRAMEBUNDLESHOULDBEOFF",
            ctypes.c_int32(0x80000834).value: "DCAMERR_INVALIDSESSIONINDEX",
            0: "DCAMERR_ERROR"}

DCAMPROP_ATTR_HASRANGE = int("0x80000000", 0)
DCAMPROP_ATTR_HASVALUETEXT = int("0x10000000", 0)
DCAMPROP_ATTR_READABLE = int("0x00010000", 0)
DCAMPROP_ATTR_WRITABLE = int("0x00020000", 0)

DCAMPROP_OPTION_NEAREST = int("0x80000000", 0)
DCAMPROP_OPTION_NEXT = int("0x01000000", 0)
DCAMPROP_OPTION_SUPPORT = int("0x00000000", 0)

DCAMPROP_TYPE_MODE = int("0x00000001", 0)
DCAMPROP_TYPE_LONG = int("0x00000002", 0)
DCAMPROP_TYPE_REAL = int("0x00000003", 0)
DCAMPROP_TYPE_MASK = int("0x0000000F", 0)

DCAMPROP_BINNING__1 = 1
DCAMPROP_BINNING__2 = 2
DCAMPROP_BINNING__4 = 4

DCAMPROP_TRIGGERSOURCE__INTERNAL = 1
DCAMPROP_TRIGGERSOURCE__EXTERNAL = 2

DCAMPROP_TRIGGER_MODE__NORMAL = 1
DCAMPROP_TRIGGER_MODE__START = 6

DCAMPROP_TRIGGERPOLARITY__NEGATIVE = 1
DCAMPROP_TRIGGERPOLARITY__POSITIVE = 2

DCAMPROP_TRIGGERACTIVE__EDGE = 1
DCAMPROP_TRIGGERACTIVE__LEVEL = 2
DCAMPROP_TRIGGERACTIVE__SYNCREADOUT = 3

DCAMPROP_OUTPUTTRIGGER_KIND__LOW = 1
DCAMPROP_OUTPUTTRIGGER_KIND__EXPOSURE = 2
DCAMPROP_OUTPUTTRIGGER_KIND__PROGRAMMABLE = 3
DCAMPROP_OUTPUTTRIGGER_KIND__TRIGGERREADY = 4
DCAMPROP_OUTPUTTRIGGER_KIND__HIGH = 5

DCAMPROP_OUTPUTTRIGGER_POLARITY__NEGATIVE = 1
DCAMPROP_OUTPUTTRIGGER_POLARITY__POSITIVE = 2

DCAMPROP_OUTPUTTRIGGER_SOURCE__READOUTEND = 2
DCAMPROP_OUTPUTTRIGGER_SOURCE__VSYNC = 3
DCAMPROP_OUTPUTTRIGGER_SOURCE__TRIGGER = 6

DCAMCAP_STATUS_ERROR = int("0x00000000", 0)
DCAMCAP_STATUS_BUSY = int("0x00000001", 0)
DCAMCAP_STATUS_READY = int("0x00000002", 0)
DCAMCAP_STATUS_STABLE = int("0x00000003", 0)
DCAMCAP_STATUS_UNSTABLE = int("0x00000004", 0)

DCAMWAIT_CAPEVENT_FRAMEREADY = int("0x0002", 0)
DCAMWAIT_CAPEVENT_STOPPED = int("0x0010", 0)

DCAMWAIT_RECEVENT_MISSED = int("0x00000200", 0)
DCAMWAIT_RECEVENT_STOPPED = int("0x00000400", 0)
DCAMWAIT_TIMEOUT_INFINITE = int("0x80000000", 0)

DCAM_DEFAULT_ARG = 0

DCAM_IDSTR_MODEL = int("0x04000104", 0)

DCAMCAP_TRANSFERKIND_FRAME = 0

DCAMCAP_START_SEQUENCE = -1
DCAMCAP_START_SNAP = 0

DCAMBUF_ATTACHKIND_FRAME = 0


# Hamamatsu structures.

## DCAMAPI_INIT
#
# The dcam initialization structure
#
class DCAMAPI_INIT(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int32),
                ("iDeviceCount", ctypes.c_int32),
                ("reserved", ctypes.c_int32),
                ("initoptionbytes", ctypes.c_int32),
                ("initoption", ctypes.POINTER(ctypes.c_int32)),
                ("guid", ctypes.POINTER(ctypes.c_int32))]


## DCAMDEV_OPEN
#
# The dcam open structure
#
class DCAMDEV_OPEN(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int32),
                ("index", ctypes.c_int32),
                ("hdcam", ctypes.c_void_p)]


## DCAMWAIT_OPEN
#
# The dcam wait open structure
#
class DCAMWAIT_OPEN(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int32),
                ("supportevent", ctypes.c_int32),
                ("hwait", ctypes.c_void_p),
                ("hdcam", ctypes.c_void_p)]


## DCAMWAIT_START
#
# The dcam wait start structure
#
class DCAMWAIT_START(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int32),
                ("eventhappened", ctypes.c_int32),
                ("eventmask", ctypes.c_int32),
                ("timeout", ctypes.c_int32)]


## DCAMCAP_TRANSFERINFO
#
# The dcam capture info structure
#
class DCAMCAP_TRANSFERINFO(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int32),
                ("iKind", ctypes.c_int32),
                ("nNewestFrameIndex", ctypes.c_int32),
                ("nFrameCount", ctypes.c_int32)]


## DCAMBUF_ATTACH
#
# The dcam buffer attachment structure
#
class DCAMBUF_ATTACH(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int32),
                ("iKind", ctypes.c_int32),
                ("buffer", ctypes.POINTER(ctypes.c_void_p)),
                ("buffercount", ctypes.c_int32)]


## DCAMBUF_FRAME
#
# The dcam buffer frame structure
#
class DCAMBUF_FRAME(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int32),
                ("iKind", ctypes.c_int32),
                ("option", ctypes.c_int32),
                ("iFrame", ctypes.c_int32),
                ("buf", ctypes.c_void_p),
                ("rowbytes", ctypes.c_int32),
                ("type", ctypes.c_int32),
                ("width", ctypes.c_int32),
                ("height", ctypes.c_int32),
                ("left", ctypes.c_int32),
                ("top", ctypes.c_int32),
                ("timestamp", ctypes.c_int32),
                ("framestamp", ctypes.c_int32),
                ("camerastamp", ctypes.c_int32)]


## DCAMDEV_STRING
#
# The dcam device string structure
#
class DCAMDEV_STRING(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int32),
                ("iString", ctypes.c_int32),
                ("text", ctypes.c_char_p),
                ("textbytes", ctypes.c_int32)]


## DCAMPROP_ATTR
#
# The dcam property attribute structure.
#
class DCAMPROP_ATTR(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_int32),
                ("iProp", ctypes.c_int32),
                ("option", ctypes.c_int32),
                ("iReserved1", ctypes.c_int32),
                ("attribute", ctypes.c_int32),
                ("iGroup", ctypes.c_int32),
                ("iUnit", ctypes.c_int32),
                ("attribute2", ctypes.c_int32),
                ("valuemin", ctypes.c_double),
                ("valuemax", ctypes.c_double),
                ("valuestep", ctypes.c_double),
                ("valuedefault", ctypes.c_double),
                ("nMaxChannel", ctypes.c_int32),
                ("iReserved3", ctypes.c_int32),
                ("nMaxView", ctypes.c_int32),
                ("iProp_NumberOfElement", ctypes.c_int32),
                ("iProp_ArrayBase", ctypes.c_int32),
                ("iPropStep_Element", ctypes.c_int32)]


class DCAMREC_OPEN(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int32),
                ("reserved", ctypes.c_int32),
                ("hrec", ctypes.c_void_p),
                ("path", ctypes.c_wchar_p),
                ("ext", ctypes.c_wchar_p),
                ("maxframepersession", ctypes.c_int32),
                ("userdatasize", ctypes.c_int32),
                ("userdatasize_session", ctypes.c_int32),
                ("userdatasize_file", ctypes.c_double),
                ("usertextsize", ctypes.c_double),
                ("usertextsize_session", ctypes.c_double),
                ("usertextsize_file", ctypes.c_double)]


class DCAMREC_STATUS(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int32),
                ("currentsession_index", ctypes.c_int32),
                ("maxframecount_per_session", ctypes.c_int32),
                ("currentframe_index", ctypes.c_int32),
                ("missingframe_count", ctypes.c_int32),
                ("flags", ctypes.c_int32),
                ("totalframecount", ctypes.c_int32),
                ("reserved", ctypes.c_int32)]


## DCAMPROP_VALUETEXT
#
# The dcam text property structure.
#
class DCAMPROP_VALUETEXT(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_int32),
                ("iProp", ctypes.c_int32),
                ("value", ctypes.c_double),
                ("text", ctypes.c_char_p),
                ("textbytes", ctypes.c_int32)]


def convertPropertyName(p_name):
    """
    "Regularizes" a property name. We are using all lowercase names with
    the spaces replaced by underscores.
    """
    return p_name.lower().replace(" ", "_")


class DCAMException(Exception):
    pass


class HCamData(object):
    """
    Hamamatsu camera data object.

    Initially I tried to use create_string_buffer() to allocate storage for the 
    data from the camera but this turned out to be too slow. The software
    kept falling behind the camera and create_string_buffer() seemed to be the
    bottleneck.

    Using numpy makes a lot more sense anyways..
    """

    def __init__(self, size=None, **kwds):
        """
        Create a data object of the appropriate size.
        """
        super().__init__(**kwds)
        self.np_array = np.ascontiguousarray(np.empty(int(size / 2), dtype=np.uint16))
        # self.np_array is a contiguous array in memory, that has size/2 elements of uint16 type I think...)
        # It's size over two since we input the number of bytes of a frame, and we reserve space for uint16 variable (bytes are 8 bit, their ratio is 2)

        self.size = size

    def __getitem__(self, slicing):
        return self.np_array[slicing]

    def copyData(self, address):
        """
        Uses the C memmove function to copy data from an address in memory
        into memory allocated for the numpy array of this object.
        """
        ctypes.memmove(self.np_array.ctypes.data, address,
                       self.size)  # copies self.size bytes from address into self.np_array.ctyps.data

    def getData(self):
        return self.np_array  # data

    def getDataPtr(self):
        return self.np_array.ctypes.data  # pointer to the address in memory of the data


class HamamatsuDevice(object):
    """
    Basic camera interface class.
    
    This version uses the Hamamatsu library to allocate camera buffers.
    Storage for the data from the camera is allocated dynamically and
    copied out of the camera buffers.
    """

    def __init__(self, frame_x, frame_y, acquisition_mode, number_frames, exposure, trsource, trmode, trpolarity,
                 tractive, ouchannel1, ouchannel2, ouchannel3, outrsource1, outrsource2, outrsource3,subarrayh_pos,
                 subarrayv_pos, binning, hardware, camera_id=None, **kwds):
        """
        Open the connection to the camera specified by camera_id.
        """
        super().__init__(**kwds)
        dcam = ctypes.windll.dcamapi
        # dcam = ctypes.windll.LoadLibrary('C:/Windows/System32/DCAMAPI/Modules/Digital/dcamapi.dll')
        paraminit = DCAMAPI_INIT(0, 0, 0, 0, None, None)
        paraminit.size = ctypes.sizeof(paraminit)
        error_code = dcam.dcamapi_init(ctypes.byref(paraminit))
        if (error_code != DCAMERR_NOERROR):
            raise DCAMException("DCAM initialization failed with error code " + str(error_code))
        """
        If the camera is not connected, you will have an error. Change?
        """
        n_cameras = paraminit.iDeviceCount
        self.dcam = dcam
        self.buffer_index = 0
        self.camera_id = camera_id
        self.debug = False
        self.encoding = 'utf-8'
        self.frame_bytes = 0
        self.frame_x = 0
        self.frame_y = 0
        self.last_frame_number = 0
        self.properties = None
        self.max_backlog = 0
        self.number_image_buffers = 0
        self.hardware = hardware  # to have a communication between hardware and device, I create this attribute
        # dictionaries for trigger properties
        self.trig_dict_source = {"internal": DCAMPROP_TRIGGERSOURCE__INTERNAL,
                                 "external": DCAMPROP_TRIGGERSOURCE__EXTERNAL}
        self.trig_dict_mode = {"normal": DCAMPROP_TRIGGER_MODE__NORMAL, "start": DCAMPROP_TRIGGER_MODE__START}
        self.trig_dict_polarity = {"negative": DCAMPROP_TRIGGERPOLARITY__NEGATIVE,
                                   "positive": DCAMPROP_TRIGGERPOLARITY__POSITIVE}
        self.trig_dict_active = {"edge": DCAMPROP_TRIGGERACTIVE__EDGE, "level": DCAMPROP_TRIGGERACTIVE__LEVEL,
                                 "syncreadout": DCAMPROP_TRIGGERACTIVE__SYNCREADOUT}
        self.trig_dict_outputtriggerkind = {"low": DCAMPROP_OUTPUTTRIGGER_KIND__LOW,
                                            "exposure": DCAMPROP_OUTPUTTRIGGER_KIND__EXPOSURE,
                                            "programmable": DCAMPROP_OUTPUTTRIGGER_KIND__PROGRAMMABLE,
                                            "trigger ready": DCAMPROP_OUTPUTTRIGGER_KIND__TRIGGERREADY,
                                            "high": DCAMPROP_OUTPUTTRIGGER_KIND__HIGH}
        self.trig_dict_outputtriggerpolarity = {"negative": DCAMPROP_OUTPUTTRIGGER_POLARITY__NEGATIVE,
                                                "positive": DCAMPROP_OUTPUTTRIGGER_POLARITY__POSITIVE}

        self.trig_dict_outputtriggersource = {"readout_end": DCAMPROP_OUTPUTTRIGGER_SOURCE__READOUTEND,
                                              "readout_start": DCAMPROP_OUTPUTTRIGGER_SOURCE__VSYNC,
                                              "input_trigger_signal": DCAMPROP_OUTPUTTRIGGER_SOURCE__TRIGGER
                                              }

        self.acquisition_mode = acquisition_mode
        self.number_frames = number_frames

        # Get camera model.
        self.camera_model = self.getModelInfo()

        # Open the camera.
        paramopen = DCAMDEV_OPEN(0, self.camera_id, None)
        paramopen.size = ctypes.sizeof(paramopen)
        self.checkStatus(self.dcam.dcamdev_open(ctypes.byref(paramopen)),
                         "dcamdev_open")
        self.camera_handle = ctypes.c_void_p(paramopen.hdcam)

        # Set up wait handle
        paramwait = DCAMWAIT_OPEN(0, 0, None, self.camera_handle)
        paramwait.size = ctypes.sizeof(paramwait)
        self.checkStatus(self.dcam.dcamwait_open(ctypes.byref(paramwait)),
                         "dcamwait_open")
        self.wait_handle = ctypes.c_void_p(paramwait.hwait)

        # Get camera properties.
        self.properties = self.getCameraProperties()
        self._addArrayCameraProperties()

        for key in self.properties:
            print(
                f'name: {key}, id: {self.properties[key]}, value: {self.getPropertyValue(key)}, '
                f'attr NumberOfElement: {self.getPropertyAttribute(key).iProp_NumberOfElement}, '
                f'attr PropStep_element: {self.getPropertyAttribute(key).iPropStep_Element}'
                f', Property value range: {self.getPropertyRange(key)}'
            )

        # Get camera max width, height.
        self.max_width = self.getPropertyValue("image_width")[0]
        self.max_height = self.getPropertyValue("image_height")[0]

        # Here we set the values in order to change these properties before the connection

        if __name__ != "__main__":
            self.setExposure(exposure)
            self.setSubarrayH(frame_x)
            self.setSubarrayV(frame_y)
            self.setSubArrayMode()
            self.setTriggerSource(trsource)
            self.setTriggerMode(trmode)
            self.setTriggerPolarity(trpolarity)
            self.setTriggerActive(tractive)
            # self.setOutputChannel1(troutput)
            self.setOutputTrigger1(ouchannel1)
            self.setOutputTrigger2(ouchannel2)
            self.setOutputTrigger3(ouchannel3)
            self.setSubarrayHpos(subarrayh_pos)
            self.setSubarrayVpos(subarrayv_pos)
            self.setBinning(binning)

    def captureSetup(self):
        """
        Capture setup (internal use only). This is called at the start
        of new acquisition sequence to determine the current ROI and
        get the camera configured properly.
        """
        self.buffer_index = -1
        self.last_frame_number = 0

        # Set sub array mode.
        self.setSubArrayMode()

        # Get frame properties.
        self.frame_x = self.getPropertyValue("image_width")[0]
        self.frame_y = self.getPropertyValue("image_height")[0]
        self.frame_bytes = self.getPropertyValue("image_framebytes")[0]

    def checkStatus(self, fn_return, fn_name="unknown", dcamproperty="unknown"):
        """
        Check return value of the dcam function call.
        Throw an error if not as expected?
        """
        # if (fn_return != DCAMERR_NOERROR) and (fn_return != DCAMERR_ERROR):
        #    raise DCAMException("dcam error: " + fn_name + " returned " + str(fn_return))
        if (fn_return in err_dict):
            c_buf_len = 80
            c_buf = ctypes.create_string_buffer(c_buf_len)
            c_error = self.dcam.dcam_getlasterror(self.camera_handle,
                                                  c_buf,
                                                  ctypes.c_int32(c_buf_len))
            # if c_buf.value in err_dict: #if the error is present in the list, we call it by name
            # raise DCAMException("dcam error in " + str(fn_name) + " ==> " + err_dict[fn_return]+ " ==> " + str(c_buf.value) )
            if dcamproperty == "unknown":
                print("dcam error in " + str(fn_name) + " ==> " + err_dict[fn_return] + " ==> " + str(c_buf.value))
            else:
                print("dcam error in " + str(fn_name) + " for " + dcamproperty + " ==> " + err_dict[
                    fn_return] + " ==> " + str(c_buf.value))
            # else:
            #   raise DCAMException("dcam error " + str(fn_name) + " " + str(c_buf.value) + "unknown error")
            # print "dcam error", fn_name, c_buf.value
        return fn_return

    def getCameraProperties(self):
        """
        Return the ids & names of all the properties that the camera supports. This
        is used at initialization to populate the self.properties attribute.
        """
        c_buf_len = 64
        c_buf = ctypes.create_string_buffer(c_buf_len)
        properties = {}
        prop_id = ctypes.c_int32(0)

        # Reset to the start.
        ret = self.dcam.dcamprop_getnextid(self.camera_handle,
                                           ctypes.byref(prop_id),
                                           ctypes.c_uint32(DCAMPROP_OPTION_NEAREST))
        if (ret != 0) and (ret != DCAMERR_NOERROR):
            self.checkStatus(ret, "dcamprop_getnextid")

        # Get the first property.
        ret = self.dcam.dcamprop_getnextid(self.camera_handle,
                                           ctypes.byref(prop_id),
                                           ctypes.c_int32(DCAMPROP_OPTION_NEXT))
        if (ret != 0) and (ret != DCAMERR_NOERROR):
            self.checkStatus(ret, "dcamprop_getnextid")
        self.checkStatus(self.dcam.dcamprop_getname(self.camera_handle,
                                                    prop_id,
                                                    c_buf,
                                                    ctypes.c_int32(c_buf_len)),
                         "dcamprop_getname")

        # Get the rest of the properties.
        last = -1
        while (prop_id.value != last):
            last = prop_id.value
            property_name = convertPropertyName(c_buf.value.decode(self.encoding))
            properties[property_name] = prop_id.value
            ret = self.dcam.dcamprop_getnextid(self.camera_handle,
                                               ctypes.byref(prop_id),
                                               ctypes.c_int32(DCAMPROP_OPTION_NEXT))
            if (ret != 0) and (ret != DCAMERR_NOERROR):
                self.checkStatus(ret, "dcamprop_getnextid")
            self.checkStatus(self.dcam.dcamprop_getname(self.camera_handle,
                                                        prop_id,
                                                        c_buf,
                                                        ctypes.c_int32(c_buf_len)),
                             "dcamprop_getname")

        return properties

    def _addArrayCameraProperties(self):
        new_properties = {}
        for key in self.properties.keys():
            if '[0]' in key:
                num_element_id = self.getPropertyAttribute(key).iProp_NumberOfElement
                num_key = next(k for k, value in self.properties.items() if value == num_element_id)
                num_element = self.getPropertyValue(num_key)[0]
                offset = self.getPropertyAttribute(key).iPropStep_Element
                for i in range(1, num_element):
                    new_key = key.replace('[0]', f'[{i}]')
                    new_properties[new_key] = self.properties[key] + i * offset
        self.properties.update(new_properties)

    def getModelInfo(self):
        """
        Returns the model of the camera
        """
        camera_id = 0  # camera_id is no more an input to make this function compatible with hardware_read_func of ScopeFoundry
        c_buf_len = 20
        string_value = ctypes.create_string_buffer(c_buf_len)
        paramstring = DCAMDEV_STRING(
            0,
            DCAM_IDSTR_MODEL,
            ctypes.cast(string_value, ctypes.c_char_p),
            c_buf_len)
        paramstring.size = ctypes.sizeof(paramstring)

        self.checkStatus(self.dcam.dcamdev_getstring(ctypes.c_int32(camera_id),
                                                     ctypes.byref(paramstring)),
                         "dcamdev_getstring")

        return string_value.value.decode(self.encoding)

    def getProperties(self):
        """
        Return the list of camera properties. This is the one to call if you
        want to know the camera properties.
        """
        return self.properties

    def getPropertyAttribute(self, property_name):
        """
        Return the attribute structure of a particular property.
        
        FIXME (OPTIMIZATION): Keep track of known attributes?
        """
        p_attr = DCAMPROP_ATTR()
        p_attr.cbSize = ctypes.sizeof(p_attr)
        p_attr.iProp = self.properties[property_name]
        ret = self.checkStatus(self.dcam.dcamprop_getattr(self.camera_handle,
                                                          ctypes.byref(p_attr)),
                               "dcamprop_getattr")
        if (ret == 0):
            print("property", property_name, "is not supported")
            return False
        else:
            return p_attr

    def getPropertyRange(self, property_name):
        """
        Return the range for an attribute.
        """
        prop_attr = self.getPropertyAttribute(property_name)
        temp = prop_attr.attribute & DCAMPROP_TYPE_MASK
        if (temp == DCAMPROP_TYPE_REAL):
            return [float(prop_attr.valuemin), float(prop_attr.valuemax)]
        else:
            return [int(prop_attr.valuemin), int(prop_attr.valuemax)]

    def getPropertyRW(self, property_name):
        """
        Return if a property is readable / writeable.
        """
        prop_attr = self.getPropertyAttribute(property_name)
        rw = []

        # Check if the property is readable.
        if (prop_attr.attribute & DCAMPROP_ATTR_READABLE):
            rw.append(True)
        else:
            rw.append(False)

        # Check if the property is writeable.
        if (prop_attr.attribute & DCAMPROP_ATTR_WRITABLE):
            rw.append(True)
        else:
            rw.append(False)

        return rw

    def getPropertyText(self, property_name):
        """
        #Return the text options of a property (if any).
        """
        prop_attr = self.getPropertyAttribute(property_name)
        if not (prop_attr.attribute & DCAMPROP_ATTR_HASVALUETEXT):
            return {}
        else:
            # Create property text structure.
            prop_id = self.properties[property_name]
            v = ctypes.c_double(prop_attr.valuemin)

            prop_text = DCAMPROP_VALUETEXT()
            c_buf_len = 64
            c_buf = ctypes.create_string_buffer(c_buf_len)
            # prop_text.text = ctypes.c_char_p(ctypes.addressof(c_buf))
            prop_text.cbSize = ctypes.c_int32(ctypes.sizeof(prop_text))
            prop_text.iProp = ctypes.c_int32(prop_id)
            prop_text.value = v
            prop_text.text = ctypes.addressof(c_buf)
            prop_text.textbytes = c_buf_len

            # Collect text options.
            done = False
            text_options = {}
            while not done:
                # Get text of current value.
                self.checkStatus(self.dcam.dcamprop_getvaluetext(self.camera_handle,
                                                                 ctypes.byref(prop_text)),
                                 "dcamprop_getvaluetext")
                text_options[prop_text.text.decode(self.encoding)] = int(v.value)

                # Get next value.
                ret = self.dcam.dcamprop_queryvalue(self.camera_handle,
                                                    ctypes.c_int32(prop_id),
                                                    ctypes.byref(v),
                                                    ctypes.c_int32(DCAMPROP_OPTION_NEXT))
                prop_text.value = v

                if (ret != 1):
                    done = True

            return text_options

    def getPropertyValue(self, property_name):
        """
        Return the current setting of a particular property.
        """

        # Check if the property exists.
        if not (property_name in self.properties):
            print(" unknown property name:", property_name)
            return False
        prop_id = self.properties[property_name]

        # Get the property attributes.
        prop_attr = self.getPropertyAttribute(property_name)

        # Get the property value.
        c_value = ctypes.c_double(0)
        self.checkStatus(self.dcam.dcamprop_getvalue(self.camera_handle,
                                                     ctypes.c_int32(prop_id),
                                                     ctypes.byref(c_value)),
                         "dcamprop_getvalue")

        # Convert type based on attribute type.
        temp = prop_attr.attribute & DCAMPROP_TYPE_MASK
        if (temp == DCAMPROP_TYPE_MODE):
            prop_type = "MODE"
            prop_value = int(c_value.value)
        elif (temp == DCAMPROP_TYPE_LONG):
            prop_type = "LONG"
            prop_value = int(c_value.value)
        elif (temp == DCAMPROP_TYPE_REAL):
            prop_type = "REAL"
            prop_value = c_value.value
        else:
            prop_type = "NONE"
            prop_value = False

        return [prop_value, prop_type]

    def getPropertiesValues(self):

        for i in self.properties:
            prop_attr = self.getPropertyValue(i)
            print("{} : {}".format(i, prop_attr[0]))

    def getTemperature(self):
        '''
        If the camera model has the temperature property value, returns the value as a string.
        Otherwise (like  with the model C11440-22CU) returns the status of the cooler.
        '''

        if self.camera_model == "C11440-22CU":
            T = 'mode ' + str(self.getPropertyValue("sensor_cooler_status")[0])
        else:
            T = self.getPropertyValue("sensor_temperature")[0]
        return T

    def isCameraProperty(self, property_name):
        """
        Check if a property name is supported by the camera.
        """
        if (property_name in self.properties):
            return True
        else:
            return False

    def setExposure(self, exposure):

        self.setPropertyValue("exposure_time", exposure)
        if self.hardware.internal_frame_rate.hardware_read_func:  # otherwise, if we have not defined yet the function, we have an error...
            self.hardware.internal_frame_rate.read_from_hardware()

    def getExposure(self):

        return self.getPropertyValue("exposure_time")[0]

    def setSubarrayH(self, hsize):

        if hsize % 4 != 0:  # If the size is not a multiple of four, is not an allowed value
            hsize = hsize - hsize % 4  # make the size a multiple of four

        """
        We must reset the value of the offset since sometimes it could happen that
        the program want to write a value of the offset while it's keeping in memory
        previous values of size, this could lead to an error if the sum of offset 
        and size overcome 2048
        """

        self.setPropertyValue("subarray_hpos", 0)
        self.setPropertyValue("subarray_hsize", hsize)

        if self.hardware.optimal_offset.val:
            self.setSubarrayHpos(self.calculateOptimalPos(int(hsize)))

    def getSubarrayH(self):

        return self.getPropertyValue("subarray_hsize")[0]

    def setSubarrayHpos(self, hpos):

        if hpos == 0:  # Necessary for not showing the below message when we are at 2048 (subarray OFF)
            self.setPropertyValue("subarray_hpos", hpos)
            return None

        if self.setSubArrayMode() == "OFF":
            print("You must be in subarray mode to change position")
            return None

        if hpos % 4 != 0:  # If the size is not a multiple of four, is not an allowed value
            hpos = hpos - hpos % 4  # make the size a multiple of four

        maximum = self.getPropertyRange("subarray_hpos")[1]  # max value
        # if vpos > 1020: #If we have 4 pixel of size, the algorithm for the optimal position fails,
        # since the max value for the offset is 1020 (while with 4 pixels it tries to write 1022)
        if hpos > maximum:
            hpos = maximum

        self.setPropertyValue("subarray_hpos", hpos)

    def getSubarrayHpos(self):

        return self.getPropertyValue("subarray_hpos")[0]

    def setSubarrayV(self, vsize):

        if vsize % 4 != 0:
            vsize = vsize - vsize % 4

        """
        We must reset the value of the offset since sometimes it could happen that
        the program want to write a value of the offset while it's keeping in memory
        previous values of size, this coulde lead to an error if the sum of offset 
        and size overcome 2048
        """

        self.setPropertyValue("subarray_vpos", 0)
        self.setPropertyValue("subarray_vsize", vsize)

        if self.hardware.optimal_offset.val:
            self.setSubarrayVpos(self.calculateOptimalPos(int(vsize)))
            self.getPropertyValue("subarray_vpos")

    def getSubarrayV(self):

        return self.getPropertyValue("subarray_vsize")[0]

    def setSubarrayVpos(self, vpos):

        if vpos == 0:  # Necessary for not showing the below message when we are at 2048 (subarray OFF)
            self.setPropertyValue("subarray_vpos", vpos)
            return None

        if self.setSubArrayMode() == "OFF":
            print("You must be in subarray mode to change position")
            return None

        if vpos % 4 != 0:  # If the size is not a multiple of four, is not an allowed value
            vpos = vpos - vpos % 4  # make the size a multiple of four

        maximum = self.getPropertyRange("subarray_vpos")[1]  # max value
        # if vpos > 1020: #If we have 4 pixel of size, the algorithm for the optimal position fails,
        # since the max value for the offset is 1020 (while with 4 pixels it tries to write 1022)
        if vpos > maximum:
            vpos = maximum

        self.setPropertyValue("subarray_vpos", vpos)

    def getSubarrayVpos(self):

        return self.getPropertyValue("subarray_vpos")[0]

    def calculateOptimalPos(self, axis_size):
        # I have found a kind of algorithm for retrieving the optimal offset from the axis size (I don't know if it is totally true)

        n = int(log2(2048 / axis_size))

        if n == 0:
            opt_pos = 0

        else:
            opt_pos = 0
            for i in range(n):
                opt_pos = opt_pos + 512 / 2 ** i

        return int(opt_pos)

    def setPropertyValue(self, property_name, property_value):
        """
        Set the value of a property.
        """

        # Check if the property exists.
        if not (property_name in self.properties):
            print(" unknown property name:", property_name)
            return False

        # If the value is text, figure out what the 
        # corresponding numerical property value is.
        if (isinstance(property_value, str)):
            text_values = self.getPropertyText(property_name)
            if (property_value in text_values):
                property_value = float(text_values[property_value])
            else:
                print(" unknown property text value:", property_value, "for", property_name)
                return False

        # Check that the property is within range.
        [pv_min, pv_max] = self.getPropertyRange(property_name)
        if (property_value < pv_min):
            print(" set property value", property_value, "is less than minimum of", pv_min, property_name,
                  "setting to minimum")
            property_value = pv_min
        if (property_value > pv_max):
            print(" set property value", property_value, "is greater than maximum of", pv_max, property_name,
                  "setting to maximum")
            property_value = pv_max

        # Set the property value, return what it was set too.
        prop_id = self.properties[property_name]
        p_value = ctypes.c_double(property_value)
        param = self.checkStatus(self.dcam.dcamprop_setgetvalue(self.camera_handle,
                                                                ctypes.c_int32(prop_id),
                                                                ctypes.byref(p_value),
                                                                ctypes.c_int32(DCAM_DEFAULT_ARG)),
                                 "dcamprop_setgetvalue", dcamproperty=property_name)
        if param == DCAMERR_INVALIDPARAM:
           actual_val = self.getPropertyValue(property_name)[0]
           raise DCAMException(" The parameter is not valid, the set value is: {}".format(actual_val))

        return p_value.value

    def setSubArrayMode(self):
        """
        This sets the sub-array mode as appropriate based on the current ROI.
        """

        # Check ROI properties.
        roi_w = self.getPropertyValue("subarray_hsize")[0]
        roi_h = self.getPropertyValue("subarray_vsize")[0]

        # If the ROI is smaller than the entire frame turn on subarray mode
        if ((roi_w == self.max_width) and (roi_h == self.max_height)):
            self.setPropertyValue("subarray_mode", "OFF")
        else:
            self.setPropertyValue("subarray_mode", "ON")

        if self.getPropertyValue("subarray_mode")[0] == 1:
            return "OFF"
        else:
            return "ON"

    def setAcquisition(self, acq_mode):
        #        self.stopAcquisition()
        self.acquisition_mode = acq_mode

    def setBinning(self, binning):

        if self.isCapturing() != DCAMCAP_STATUS_BUSY:
            self.setPropertyValue("binning", binning)

    def getBinning(self):

        return self.getPropertyValue("binning")[0]

    def setNumberImages(self, num_images):
        #       self.stopAcquisition()
        if num_images < 1:
            print("The number of frames can't be less than 1.")
            return None
        else:
            self.number_frames = num_images

    def setACQMode(self, mode, number_frames=None):
        '''
        Set the acquisition mode to either run until aborted or to 
        stop after acquiring a set number of frames.

        mode should be either "fixed_length" or "run_till_abort"

        if mode is "fixed_length", then number_frames indicates the number
        of frames to acquire.
        '''

        self.stopAcquisition()

        if self.acquisition_mode == "fixed_length" or \
                self.acquisition_mode == "run_till_abort":
            self.acquisition_mode = mode
            self.number_frames = number_frames
        else:
            raise DCAMException("Unrecognised acquisition mode: " + mode)

    def setTriggerSource(self, trsource):

        if self.isCapturing() != DCAMCAP_STATUS_BUSY:
            self.setPropertyValue("trigger_source", self.trig_dict_source[trsource])

    def setTriggerMode(self, trmode):

        if self.isCapturing() != DCAMCAP_STATUS_BUSY:
            self.setPropertyValue("trigger_mode", self.trig_dict_mode[trmode])

    def setTriggerPolarity(self, trpolarity):

        if self.isCapturing() != DCAMCAP_STATUS_BUSY:
            self.setPropertyValue("trigger_polarity", self.trig_dict_polarity[trpolarity])

    def setTriggerActive(self, tractive):

        if self.isCapturing() != DCAMCAP_STATUS_BUSY:
            self.setPropertyValue("trigger_active", self.trig_dict_active[tractive])

    def setOutputTrigger1Polarity(self, outrpolarity1):

        if self.isCapturing() != DCAMCAP_STATUS_BUSY:
            self.setPropertyValue("output_trigger_polarity[0]", self.trig_dict_outputtriggerpolarity[outrpolarity1])

    def setOutputTrigger2Polarity(self, outrpolarity2):

        if self.isCapturing() != DCAMCAP_STATUS_BUSY:
            self.setPropertyValue("output_trigger_polarity[1]", self.trig_dict_outputtriggerpolarity[outrpolarity2])

    def setOutputTrigger3Polarity(self, outrpolarity3):

        if self.isCapturing() != DCAMCAP_STATUS_BUSY:
            self.setPropertyValue("output_trigger_polarity[2]", self.trig_dict_outputtriggerpolarity[outrpolarity3])

    def setOutputTrigger1Source(self, outrsource1):

        if self.isCapturing() != DCAMCAP_STATUS_BUSY:
            self.setPropertyValue("output_trigger_source[0]", self.trig_dict_outputtriggersource[outrsource1])

    def setOutputTrigger2Source(self, outrsource2):

        if self.isCapturing() != DCAMCAP_STATUS_BUSY:
            self.setPropertyValue("output_trigger_source[1]", self.trig_dict_outputtriggersource[outrsource2])

    def setOutputTrigger3Source(self, outrsource3):

        if self.isCapturing() != DCAMCAP_STATUS_BUSY:
            self.setPropertyValue("output_trigger_source[2]", self.trig_dict_outputtriggersource[outrsource3])

    # ALL output trigger polarities are set as positive for convenience
    def setOutputTrigger1(self, ouchannel1):

        if self.isCapturing() != DCAMCAP_STATUS_BUSY:
            self.setPropertyValue("output_trigger_kind[0]", self.trig_dict_outputtriggerkind[ouchannel1])

        self.setOutputTrigger1Polarity("positive")

    def setOutputTrigger2(self, ouchannel2):

        if self.isCapturing() != DCAMCAP_STATUS_BUSY:
            self.setPropertyValue("output_trigger_kind[1]", self.trig_dict_outputtriggerkind[ouchannel2])

        self.setOutputTrigger2Polarity("positive")

    def setOutputTrigger3(self, ouchannel3):

        if self.isCapturing() != DCAMCAP_STATUS_BUSY:
            self.setPropertyValue("output_trigger_kind[2]", self.trig_dict_outputtriggerkind[ouchannel3])

        self.setOutputTrigger3Polarity("positive")

    def getTriggerSource(self):

        inv_dict = {v: k for k, v in self.trig_dict_source.items()}

        return inv_dict[self.getPropertyValue("trigger_source")[0]]

    def getTriggerMode(self):

        inv_dict = {v: k for k, v in self.trig_dict_mode.items()}

        return inv_dict[self.getPropertyValue("trigger_mode")[0]]

    def getTriggerPolarity(self):

        inv_dict = {v: k for k, v in self.trig_dict_polarity.items()}

        return inv_dict[self.getPropertyValue("trigger_polarity")[0]]

    def getTriggerActive(self):

        inv_dict = {v: k for k, v in self.trig_dict_active.items()}

        return inv_dict[self.getPropertyValue("trigger_active")[0]]

    def getOutputTrigger1Source(self):

        inv_dict = {v: k for k, v in self.trig_dict_outputtriggersource.items()}

        return inv_dict[self.getPropertyValue("output_trigger_source[0]")[0]]

    def getOutputTrigger2Source(self):

        inv_dict = {v: k for k, v in self.trig_dict_outputtriggersource.items()}

        return inv_dict[self.getPropertyValue("output_trigger_source[1]")[0]]

    def getOutputTrigger3Source(self):

        inv_dict = {v: k for k, v in self.trig_dict_outputtriggersource.items()}

        return inv_dict[self.getPropertyValue("output_trigger_source[2]")[0]]

    def getOutputTrigger1(self):

        inv_dict = {v: k for k, v in self.trig_dict_outputtriggerkind.items()}

        return inv_dict[self.getPropertyValue("output_trigger_kind[0]")[0]]

    def getOutputTrigger2(self):

        inv_dict = {v: k for k, v in self.trig_dict_outputtriggerkind.items()}

        return inv_dict[self.getPropertyValue("output_trigger_kind[1]")[0]]

    def getOutputTrigger3(self):

        inv_dict = {v: k for k, v in self.trig_dict_outputtriggerkind.items()}

        return inv_dict[self.getPropertyValue("output_trigger_kind[2]")[0]]

    def getOutputTrigger1Polarity(self):

        inv_dict = {v: k for k, v in self.trig_dict_outputtriggerpolarity.items()}

        return inv_dict[self.getPropertyValue("output_trigger_polarity[0]")[0]]

    def getOutputTrigger2Polarity(self):

        inv_dict = {v: k for k, v in self.trig_dict_outputtriggerpolarity.items()}

        return inv_dict[self.getPropertyValue("output_trigger_polarity[1]")[0]]

    def getOutputTrigger3Polarity(self):

        inv_dict = {v: k for k, v in self.trig_dict_outputtriggerpolarity.items()}

        return inv_dict[self.getPropertyValue("output_trigger_polarity[2]")[0]]

    def isCapturing(self):

        captureStatus = ctypes.c_int32(0)
        self.checkStatus(self.dcam.dcamcap_status(
            self.camera_handle, ctypes.byref(captureStatus)))

        return captureStatus.value

    def getInternalFrameRate(self):

        return self.getPropertyValue("internal_frame_rate")[0]

    def getTransferInfo(self):

        captureStatus = ctypes.c_int32(0)
        self.checkStatus(self.dcam.dcamcap_status(
            self.camera_handle, ctypes.byref(captureStatus)))

        # Wait for a new frame if the camera is acquiring.
        if captureStatus.value == DCAMCAP_STATUS_BUSY:
            paramstart = DCAMWAIT_START(
                0,
                0,
                DCAMWAIT_CAPEVENT_FRAMEREADY | DCAMWAIT_CAPEVENT_STOPPED,
                DCAMWAIT_TIMEOUT_INFINITE)  # 1000 is the timeout. Remember it when changin the tmie exposure
            paramstart.size = ctypes.sizeof(paramstart)
            self.checkStatus(self.dcam.dcamwait_start(self.wait_handle,
                                                      ctypes.byref(paramstart)),
                             "dcamwait_start")

        # Check how many new frames there are.
        paramtransfer = DCAMCAP_TRANSFERINFO(
            0, DCAMCAP_TRANSFERKIND_FRAME, 0, 0)
        paramtransfer.size = ctypes.sizeof(paramtransfer)
        self.checkStatus(self.dcam.dcamcap_transferinfo(self.camera_handle,
                                                        ctypes.byref(paramtransfer)),
                         "dcamcap_transferinfo")

        """
        Put also the backlog in transfer info
        """

        cur_buffer_index = paramtransfer.nNewestFrameIndex
        cur_frame_number = paramtransfer.nFrameCount

        self.backlog = cur_frame_number - self.last_frame_number
        if (self.backlog > self.number_image_buffers):
            print(">> Warning! hamamatsu camera frame buffer overrun detected!")
        if (self.backlog > self.max_backlog):
            self.max_backlog = self.backlog
        self.last_frame_number = cur_frame_number

        return cur_buffer_index, cur_frame_number

    def newFrames(self):
        """
        Return a list of the ids of all the new frames since the last check.
        Returns an empty list if the camera has already stopped and no frames
        are available.
    
        This will block waiting for at least one new frame.
        """

        cur_buffer_index, cur_frame_number = self.getTransferInfo()

        # Check that we have not acquired more frames than we can store in our buffer.
        # Keep track of the maximum backlog.
        #         backlog = cur_frame_number - self.last_frame_number
        #         if (backlog > self.number_image_buffers):
        #             print(">> Warning! hamamatsu camera frame buffer overrun detected!")
        #         if (backlog > self.max_backlog):
        #             self.max_backlog = backlog
        #         self.last_frame_number = cur_frame_number

        # Create a list of the new frames.
        new_frames = []

        if (
                cur_buffer_index < self.buffer_index):  # this condition is mainly "False" but sometimes is true, I think when the buffer finishes its space
            for i in range(self.buffer_index + 1,
                           self.number_image_buffers):  # I need to take all the images that were in the remaining buffer
                new_frames.append(i)
            for i in range(
                    cur_buffer_index + 1):  # since the space on the buffer is finished, I restart (cur_index has been "reset")
                new_frames.append(i)
        else:  # executed the vast majority of time
            for i in range(self.buffer_index, cur_buffer_index):
                new_frames.append(i + 1)
        self.buffer_index = cur_buffer_index

        if self.debug:
            print(new_frames)

        return new_frames

    def lastTotFrames(self):
        """
        Return a list of the ids of all the new frames since the last check.
        Returns an empty list if the camera has already stopped and no frames
        are available.
    
        This will block waiting for at least one new frame.
        """

        frames = []
        """
        Pay attention! With the below code we are inserting in frames the indexes of the temporally last
        acquired images, considering the images in the whole buffer. In this way there is the risk that some
        imaages could have been overwritten when the images are fetched from the camera.
        """

        for i in range(self.buffer_index + 1, self.number_image_buffers):
            frames.append(i)

        for i in range(0, self.buffer_index + 1):
            frames.append(i)

        # if self.buffer_index > number:
        #     for i in reversed(range(0, self.buffer_index)):
        #         frames.append(i)
        # else:
        #     for i in reversed(range(0, self.number_image_buffers)):
        #         frames.append(i)
        #         if len(frames) >= number:
        #             break
        #
        # if self.debug:
        #     print(new_frames)

        return frames

    def lastEvenFrames(self):

        even_frames = []

        for i in range(self.buffer_index + 1, self.number_image_buffers):
            if i % 2 == 0:
                even_frames.append(i)

        for i in range(0, self.buffer_index + 1):
            if i % 2 == 0:
                even_frames.append(i)

        return even_frames

    def lastOddFrames(self):

        odd_frames = []

        for i in range(self.buffer_index + 1, self.number_image_buffers):
            if i % 2 != 0:
                odd_frames.append(i)

        for i in range(0, self.buffer_index + 1):
            if i % 2 != 0:
                odd_frames.append(i)

        return odd_frames

    def lastFrame(self):

        """
        Equal to lastFrames, but we only return the index of the last frame.
        """

        cur_buffer_index, cur_frame_number = self.getTransferInfo()

        # Check that we have not acquired more frames than we can store in our buffer.
        # Keep track of the maximum backlog.
        #         backlog = cur_frame_number - self.last_frame_number
        #         if (backlog > self.number_image_buffers):
        #             print(">> Warning! hamamatsu camera frame buffer overrun detected!")
        #         if (backlog > self.max_backlog):
        #             self.max_backlog = backlog
        #         self.last_frame_number = cur_frame_number

        # Create a list of the new frames.
        last_frame_index = cur_buffer_index
        self.buffer_index = cur_buffer_index

        return last_frame_index

    def getFrames(self):

        """
        Gets all of the available frames.
    
        This will block waiting for new frames even if 
        there new frames available when it is called.
        """
        frames = []
        for n in self.newFrames():
            paramlock = DCAMBUF_FRAME(
                0, 0, 0, n, None, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            paramlock.size = ctypes.sizeof(paramlock)

            # Lock the frame in the camera buffer & get address.
            self.checkStatus(self.dcam.dcambuf_lockframe(self.camera_handle,
                                                         ctypes.byref(paramlock)),
                             "dcambuf_lockframe")

            # Create storage for the frame & copy into this storage.
            hc_data = HCamData(self.frame_bytes)
            hc_data.copyData(paramlock.buf)

            frames.append(hc_data)

        return [frames, [self.frame_x, self.frame_y]]

    def getLastFrame(self):
        """
        Gets only the last frame available.
    
        """

        # frames = []

        n = self.lastFrame()

        paramlock = DCAMBUF_FRAME(
            0, 0, 0, n, None, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        paramlock.size = ctypes.sizeof(paramlock)

        # Lock the frame in the camera buffer & get address.
        self.checkStatus(self.dcam.dcambuf_lockframe(self.camera_handle,
                                                     ctypes.byref(paramlock)),
                         "dcambuf_lockframe")

        # Create storage for the frame & copy into this storage.
        hc_data = HCamData(self.frame_bytes)
        hc_data.copyData(paramlock.buf)

        frames = hc_data

        return [frames, [self.frame_x, self.frame_y]]

    def getLastTotFrames(self):
        """
        Gets the last frames in the buffer
        """
        frames = []

        for n in self.lastTotFrames():
            paramlock = DCAMBUF_FRAME(
                0, 0, 0, n, None, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            paramlock.size = ctypes.sizeof(paramlock)

            # Lock the frame in the camera buffer & get address.
            self.checkStatus(self.dcam.dcambuf_lockframe(self.camera_handle,
                                                         ctypes.byref(paramlock)),
                             "dcambuf_lockframe")

            # Create storage for the frame & copy into this storage.
            hc_data = HCamData(self.frame_bytes)
            hc_data.copyData(paramlock.buf)

            frames.append(hc_data)

        return [frames, [self.frame_x, self.frame_y]]

    def getLastEvenFrames(self):

        frames = []

        for n in self.lastEvenFrames():
            paramlock = DCAMBUF_FRAME(
                0, 0, 0, n, None, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            paramlock.size = ctypes.sizeof(paramlock)

            # Lock the frame in the camera buffer & get address.
            self.checkStatus(self.dcam.dcambuf_lockframe(self.camera_handle,
                                                         ctypes.byref(paramlock)),
                             "dcambuf_lockframe")

            # Create storage for the frame & copy into this storage.
            hc_data = HCamData(self.frame_bytes)
            hc_data.copyData(paramlock.buf)

            frames.append(hc_data)

        return [frames, [self.frame_x, self.frame_y]]

    def getLastOddFrames(self):

        frames = []

        for n in self.lastOddFrames():
            paramlock = DCAMBUF_FRAME(
                0, 0, 0, n, None, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            paramlock.size = ctypes.sizeof(paramlock)

            # Lock the frame in the camera buffer & get address.
            self.checkStatus(self.dcam.dcambuf_lockframe(self.camera_handle,
                                                         ctypes.byref(paramlock)),
                             "dcambuf_lockframe")

            # Create storage for the frame & copy into this storage.
            hc_data = HCamData(self.frame_bytes)
            hc_data.copyData(paramlock.buf)

            frames.append(hc_data)

        return [frames, [self.frame_x, self.frame_y]]

    def getRequiredFrame(self, required_index):

        """
        Gets the frame at the required index
        """

        n = required_index

        paramlock = DCAMBUF_FRAME(
            0, 0, 0, n, None, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        paramlock.size = ctypes.sizeof(paramlock)

        # Lock the frame in the camera buffer & get address.
        self.checkStatus(self.dcam.dcambuf_lockframe(self.camera_handle,
                                                     ctypes.byref(paramlock)),
                         "dcambuf_lockframe")

        # Create storage for the frame & copy into this storage.
        hc_data = HCamData(self.frame_bytes)
        hc_data.copyData(paramlock.buf)

        frames = hc_data

        return [frames, [self.frame_x, self.frame_y]]

    def startAcquisition(self):
        """
        Start data acquisition.
        """
        self.captureSetup()

        # Allocate Hamamatsu image buffers.
        # We allocate enough to buffer 2 seconds of data or the specified 
        # number of frames for a fixed length acquisition

        if self.acquisition_mode == "run_till_abort":
            # n_buffers = int(20.0*self.getPropertyValue("internal_frame_rate")[0])
            n_buffers = self.number_frames


        elif self.acquisition_mode == "fixed_length":
            n_buffers = self.number_frames

        self.number_image_buffers = n_buffers

        self.checkStatus(self.dcam.dcambuf_alloc(self.camera_handle,
                                                 ctypes.c_int32(self.number_image_buffers)),
                         "dcambuf_alloc")

        # Start acquisition.
        if self.acquisition_mode == "run_till_abort":
            self.checkStatus(self.dcam.dcamcap_start(self.camera_handle,
                                                     DCAMCAP_START_SEQUENCE),
                             "dcamcap_start")

        if self.acquisition_mode == "fixed_length":
            self.checkStatus(self.dcam.dcamcap_start(self.camera_handle,
                                                     DCAMCAP_START_SNAP),
                             "dcamcap_start")

    def startAcquisitionWithoutAlloc(self):
        """
        Start data acquisition.
        """

        # Allocate Hamamatsu image buffers.
        # We allocate enough to buffer 2 seconds of data or the specified 
        # number of frames for a fixed length acquisition

        if self.acquisition_mode == "run_till_abort":
            # n_buffers = int(20.0*self.getPropertyValue("internal_frame_rate")[0])
            n_buffers = self.number_frames


        elif self.acquisition_mode == "fixed_length":
            n_buffers = self.number_frames

        self.number_image_buffers = n_buffers

        # Start acquisition.
        if self.acquisition_mode == "run_till_abort":
            self.checkStatus(self.dcam.dcamcap_start(self.camera_handle,
                                                     DCAMCAP_START_SEQUENCE),
                             "dcamcap_start")

        if self.acquisition_mode == "fixed_length":
            self.checkStatus(self.dcam.dcamcap_start(self.camera_handle,
                                                     DCAMCAP_START_SNAP),
                             "dcamcap_start")

    def stopAcquisition(self):
        """
        Stop data acquisition.
        """

        # Stop acquisition.
        self.checkStatus(self.dcam.dcamcap_stop(self.camera_handle),
                         "dcamcap_stop")

        # print("max camera backlog was", self.max_backlog, "of", self.number_image_buffers)
        self.max_backlog = 0

        # Free image buffers.
        self.number_image_buffers = 0
        self.checkStatus(self.dcam.dcambuf_release(self.camera_handle,
                                                   DCAMBUF_ATTACHKIND_FRAME),
                         "dcambuf_release")

    def stopAcquisitionNotReleasing(self):
        # Stop acquisition.
        self.checkStatus(self.dcam.dcamcap_stop(self.camera_handle),
                         "dcamcap_stop")

        # print("max camera backlog was", self.max_backlog, "of", self.number_image_buffers)
        self.max_backlog = 0

    def releaseBuffer(self):

        self.checkStatus(self.dcam.dcambuf_release(self.camera_handle,
                                                   DCAMBUF_ATTACHKIND_FRAME),
                         "dcambuf_release")

    def shutdown(self):
        """
        Close down the connection to the camera.
        """
        self.checkStatus(self.dcam.dcamwait_close(self.wait_handle),
                         "dcamwait_close")
        self.checkStatus(self.dcam.dcamdev_close(self.camera_handle),
                         "dcamdev_close")

    def sortedPropertyTextOptions(self, property_name):
        """
        Returns the property text options a list sorted by value.
        """
        text_values = self.getPropertyText(property_name)
        return sorted(text_values, key=text_values.get)

    def startRecording(self):
        '''  Starts recording session, that will acquire self.number_frames images. Data are saved in the directory specified with the widget. '''

        # ACCESS IMAGE DATA
        # During a recording session, the host software can access the frames that have already been recorded by using the dcamrec_lockframe() or dcamrec_copyframe() function.
        # These functions will cause some stress to the computer so we do not recommend using them during a recording. 
        self.captureSetup()
        self.number_image_buffers = self.number_frames

        paramrec = DCAMREC_OPEN(0, 0, None, None, None, 0, 0, 0, 0, 0, 0, 0)

        paramrec.size = ctypes.sizeof(paramrec)
        paramrec.path = ctypes.c_wchar_p(self.hardware.app.settings.save_dir.val + "\\" + str(
            time.strftime("%Y%m%d_%H%M%S_")) + self.hardware.app.settings.sample.val)
        paramrec.ext = ctypes.c_wchar_p("dcimg")
        paramrec.maxframepersession = self.number_frames  # number of frames acquired in 1 session?
        # To use the disk recorder, the target file must be prepared first by calling the dcamrec_open() function.
        # The HDCAM handle is not used with this function. To start recording, the dcamcap_record() function should be called during READY state.
        # Finally, calling the dcamcap_start() function after dcamcap_record() starts the recording
        self.checkStatus(self.dcam.dcamrec_openW(ctypes.byref(paramrec)), "dcamrec_openW")
        self.rec_handle = ctypes.c_void_p(paramrec.hrec)

        self.checkStatus(self.dcam.dcambuf_alloc(self.camera_handle,
                                                 ctypes.c_int32(self.number_image_buffers)),
                         "dcambuf_alloc")

        self.checkStatus(self.dcam.dcamcap_record(self.camera_handle,
                                                  self.rec_handle),
                         "dcamcap_record")

        self.checkStatus(self.dcam.dcamcap_start(self.camera_handle,
                                                 DCAMCAP_START_SNAP),
                         "dcamcap_start")

    def stopRecording(self):
        '''  Waits  until capturing event stopped, then closes the recording session '''

        captureStatus = ctypes.c_int32(0)
        self.checkStatus(self.dcam.dcamcap_status(
            self.camera_handle, ctypes.byref(captureStatus)), "dcamcap_status")

        if captureStatus.value == DCAMCAP_STATUS_BUSY:
            paramstart = DCAMWAIT_START(
                0,
                0,
                DCAMWAIT_CAPEVENT_STOPPED,
                DCAMWAIT_TIMEOUT_INFINITE)  # 1000 is the timeout. Remember it when changin the tmie exposure
            paramstart.size = ctypes.sizeof(paramstart)
            self.checkStatus(self.dcam.dcamwait_start(self.wait_handle, ctypes.byref(paramstart)), "dcamwait_start")

        self.checkStatus(self.dcam.dcamcap_stop(self.camera_handle), "dcamcap_stop")

        # When the host software calls the dcamrec_close() function, all of the file information is stored.
        # Be aware that if the host software terminates without calling the dcamrec_close() function, some data may be lost.
        # print(self.checkStatus(dcam.dcamcap_stop(self.camera_handle),"dcamcap_stop"))
        self.checkStatus(self.dcam.dcamrec_close(self.rec_handle), "dcamrec_close")


# ======================================================================================================================================================

class HamamatsuDeviceMR(HamamatsuDevice):
    """
    Memory recycling camera class.
    
    This version allocates "user memory" for the Hamamatsu camera 
    buffers. This memory is also the location of the storage for
    the np_array element of a HCamData() class. The memory is
    allocated once at the beginning, then recycled. This means
    that there is a lot less memory allocation & shuffling compared
    to the basic class, which performs one allocation and (I believe)
    two copies for each frame that is acquired.
    
    WARNING: There is the potential here for chaos. Since the memory
             is now shared there is the possibility that downstream code
             will try and access the same bit of memory at the same time
             as the camera and this could end badly.

    FIXME: Use lockbits (and unlockbits) to avoid memory clashes?
           This would probably also involve some kind of reference 
           counting scheme.
    """

    def __init__(self, **kwds):
        super().__init__(**kwds)

        self.hcam_data = []
        self.hcam_ptr = False
        self.old_frame_bytes = -1

        # self.setPropertyValue("output_trigger_kind[0]", 2)

    def lastFrame(self):
        """
        Return a list of the ids of all the new frames since the last check.
        Returns an empty list if the camera has already stopped and no frames
        are available.
    
        This will block waiting for at least one new frame.
        """

        captureStatus = ctypes.c_int32(0)
        self.checkStatus(self.dcam.dcamcap_status(
            self.camera_handle, ctypes.byref(captureStatus)))

        # Wait for a new frame if the camera is acquiring.
        if captureStatus.value == DCAMCAP_STATUS_BUSY:
            paramstart = DCAMWAIT_START(
                0,
                0,
                DCAMWAIT_CAPEVENT_FRAMEREADY | DCAMWAIT_CAPEVENT_STOPPED,
                DCAMWAIT_TIMEOUT_INFINITE)  # 1000 is the timeout. Remember it when changin the tmie exposure
            paramstart.size = ctypes.sizeof(paramstart)
            self.checkStatus(self.dcam.dcamwait_start(self.wait_handle,
                                                      ctypes.byref(paramstart)),
                             "dcamwait_start")

        # Check how many new frames there are.
        paramtransfer = DCAMCAP_TRANSFERINFO(
            0, DCAMCAP_TRANSFERKIND_FRAME, 0, 0)
        paramtransfer.size = ctypes.sizeof(paramtransfer)
        self.checkStatus(self.dcam.dcamcap_transferinfo(self.camera_handle,
                                                        ctypes.byref(paramtransfer)),
                         "dcamcap_transferinfo")
        cur_buffer_index = paramtransfer.nNewestFrameIndex

        cur_frame_number = paramtransfer.nFrameCount

        # Check that we have not acquired more frames than we can store in our buffer.
        # Keep track of the maximum backlog.
        backlog = cur_frame_number - self.last_frame_number
        # if (backlog > self.number_image_buffers):
        # print(">> Warning! hamamatsu camera frame buffer overrun detected!")
        if (backlog > self.max_backlog):
            self.max_backlog = backlog
        self.last_frame_number = cur_frame_number

        # Create a list of the new frames.
        new_last_frame = cur_buffer_index
        #         if (cur_buffer_index < self.buffer_index):
        #             for i in range(self.buffer_index + 1, self.number_image_buffers):
        #                 new_frames.append(i)
        #             for i in range(cur_buffer_index + 1):
        #                 new_frames.append(i)
        #         else:
        #             for i in range(self.buffer_index, cur_buffer_index):
        #                 new_frames.append(i+1)
        self.buffer_index = cur_buffer_index

        # if self.debug:
        #    print(new_last_frame)

        # return new_last_frame

    def getFrames(self):
        """
        Gets all of the available frames.
        
        This will block waiting for new frames even if there new frames 
        available when it is called.
        
        FIXME: It does not always seem to block? The length of frames can
               be zero. Are frames getting dropped? Some sort of race condition?
        """
        frames = []
        for n in self.newFrames():
            frames.append(self.hcam_data[n])

        return [frames, [self.frame_x, self.frame_y]]

    def getLastFrame(self):

        self.lastFrame()
        frame = self.hcam_data[0]

        return [frame, [self.frame_x, self.frame_y]]

    def startAcquisition(self):
        """
        Allocate as many frames as will fit in 2GB of memory and start data acquisition.
        """
        self.captureSetup()

        # Allocate new image buffers if necessary. This will allocate
        # as many frames as can fit in 2GB of memory, or 2000 frames,
        # which ever is smaller. The problem is that if the frame size
        # is small than a lot of buffers can fit in 2GB. Assuming that
        # the camera maximum speed is something like 1KHz 2000 frames
        # should be enough for 2 seconds of storage, which will hopefully
        # be long enough.
        #
        # backslash is used to escape the newline
        if (self.old_frame_bytes != self.frame_bytes) and (self.acquisition_mode != "run_till_abort") or \
                (self.acquisition_mode == "fixed_length"):

            # n_buffers = min(int((2.0 * 1024 * 1024 * 1024)/self.frame_bytes), 2000)

            self.number_image_buffers = self.number_frames

            # Allocate new image buffers.
            ptr_array = ctypes.c_void_p * self.number_image_buffers  # crea un array del tipo c_void_p
            self.hcam_ptr = ptr_array()
            self.hcam_data = []
            for i in range(self.number_image_buffers):
                hc_data = HCamData(self.frame_bytes)
                self.hcam_ptr[i] = hc_data.getDataPtr()
                self.hcam_data.append(hc_data)

            self.old_frame_bytes = self.frame_bytes

        else:

            n_buffers = 1
            self.number_image_buffers = n_buffers
            ptr_array = ctypes.c_void_p * self.number_image_buffers  # creates a type c_void_p array
            self.hcam_ptr = ptr_array()
            self.hcam_data = []
            hc_data = HCamData(self.frame_bytes)
            self.hcam_ptr[0] = hc_data.getDataPtr()
            self.hcam_data.append(hc_data)

            self.old_frame_bytes = self.frame_bytes
        # Attach image buffers and start acquisition.
        #
        # We need to attach & release for each acquisition otherwise
        # we'll get an error if we try to change the ROI in any way
        # between acquisitions.

        paramattach = DCAMBUF_ATTACH(0, DCAMBUF_ATTACHKIND_FRAME,
                                     self.hcam_ptr, self.number_image_buffers)
        paramattach.size = ctypes.sizeof(paramattach)

        if self.acquisition_mode == "run_till_abort":
            self.checkStatus(self.dcam.dcambuf_attach(self.camera_handle,
                                                      paramattach),
                             "dcam_attachbuffer")
            self.checkStatus(self.dcam.dcamcap_start(self.camera_handle,
                                                     DCAMCAP_START_SEQUENCE),
                             "dcamcap_start")
        if self.acquisition_mode == "fixed_length":
            paramattach.buffercount = self.number_frames
            self.checkStatus(self.dcam.dcambuf_attach(self.camera_handle,
                                                      paramattach),
                             "dcambuf_attach")
            self.checkStatus(self.dcam.dcamcap_start(self.camera_handle,
                                                     DCAMCAP_START_SNAP),
                             "dcamcap_start")

    def stopAcquisition(self):
        """
        Stop data acquisition and release the memory associates with the frames.
        """

        # Stop acquisition.
        self.checkStatus(self.dcam.dcamcap_stop(self.camera_handle),
                         "dcamcap_stop")

        # Release image buffers.
        if (self.hcam_ptr):
            self.checkStatus(self.dcam.dcambuf_release(self.camera_handle,
                                                       DCAMBUF_ATTACHKIND_FRAME),
                             "dcambuf_release")

        # print("max camera backlog was:", self.max_backlog)
        self.max_backlog = 0


# if __name__ == "__main__":
#
#     import sys
#     import pyqtgraph as pg
#     import qtpy
#     from qtpy.QtWidgets import QApplication
#
#     hamamatsu = HamamatsuDevice(camera_id=0, frame_x=2048, frame_y=2048, acquisition_mode="fixed_length",
#                                 number_frames=1, exposure=0.01,
#                                 trsource="internal", trmode="normal", trpolarity="positive", tractive="edge",
#                                 ouchannel1="low", ouchannel2="low", ouchannel3="low",
#                                 subarrayh_pos=0, subarrayv_pos=0,
#                                 binning=1, hardware=None)
#     # print("found: {} cameras".format(n_cameras))
#     print("camera 0 model:", hamamatsu.getModelInfo())
#     print(type(hamamatsu.getModelInfo()))
#     print("=====================")
#     print(hamamatsu.getPropertiesValues())
#
#     hamamatsu.startAcquisition()
#     [frame, dims] = hamamatsu.getLastFrame()
#     np_data = frame.getData()
#     pg.image(np.reshape(np_data, (2048, 2048)).T)
#     hamamatsu.stopAcquisition()
#     hamamatsu.shutdown()
#     if sys.flags.interactive != 1 or not hasattr(qtpy.QtCore, 'PYQT_VERSION'):
#         QApplication.exec_()
#
# The MIT License
#
# Copyright (c) 2013 Zhuang Lab, Harvard University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#

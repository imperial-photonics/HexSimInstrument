"""
Intro:  Hardware interface of Hamamatsu camera.
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

from ScopeFoundry import HardwareComponent
from hardware.CameraHardware import *
# import Hamamatsu_ScopeFoundry.CameraDevice
from devices.CameraDevice import HamamatsuDevice, HamamatsuDeviceMR, DCAMERR_NOERROR, DCAMException


# from Hamamatsu_ScopeFoundry.CameraDevice import HamamatsuDevice, HamamatsuDeviceMR, DCAMERR_NOERROR, DCAMException


class HamamatsuHardware(HardwareComponent):
    name = "HamamatsuHardware"

    def setup(self):

        self.camera = self.add_logged_quantity('camera', dtype=str, si=False,
                                               ro=1, initial='No camera found')

        self.temperature = self.add_logged_quantity('temperature ' + chr(176) + 'C', dtype=str, si=False, ro=1)

        self.exposure_time = self.add_logged_quantity('exposure_time', dtype=float, si=False, ro=0,
                                                      spinbox_step=0.001, spinbox_decimals=3, initial=0.001, unit='s',
                                                      reread_from_hardware_after_write=True,
                                                      vmin=0)

        self.internal_frame_rate = self.add_logged_quantity('internal_frame_rate', dtype=float, si=False, ro=1,
                                                            initial=0, unit='fps')

        self.acquisition_mode = self.add_logged_quantity('acquisition_mode', dtype=str, ro=0,
                                                         choices=["fixed_length", "run_till_abort"],
                                                         initial="run_till_abort")

        self.number_frames = self.add_logged_quantity("number_frames", dtype=int, si=False, ro=0,
                                                      initial=200, vmin=1)

        # For subarray we have imposed float, since otherwise I cannot modify the step (I should modify the logged quantities script, but I prefer left it untouched)
        self.subarrayh = self.add_logged_quantity("subarray_hsize", dtype=float, si=False, ro=0,
                                                  spinbox_step=4, spinbox_decimals=0, initial=512, vmin=4, vmax=2304,
                                                  reread_from_hardware_after_write=True)

        self.subarrayv = self.add_logged_quantity("subarray_vsize", dtype=float, si=False, ro=0,
                                                  spinbox_step=4, spinbox_decimals=0, initial=512, vmin=4, vmax=2304,
                                                  reread_from_hardware_after_write=True)

        self.submode = self.add_logged_quantity("subarray_mode", dtype=str, si=False, ro=1,
                                                initial='ON')

        self.subarrayh_pos = self.add_logged_quantity('subarrayh_pos', dtype=float, si=False, ro=0,
                                                      spinbox_step=4, spinbox_decimals=0, initial=1150, vmin=0,
                                                      vmax=2304, reread_from_hardware_after_write=True,
                                                      description="The default value 0 corresponds to the first pixel starting from the left")

        self.subarrayv_pos = self.add_logged_quantity('subarrayv_pos', dtype=float, si=False, ro=0,
                                                      spinbox_step=4, spinbox_decimals=0, initial=1150, vmin=0,
                                                      vmax=2304, reread_from_hardware_after_write=True,
                                                      description="The default value 0 corresponds to the first pixel starting from the top")

        self.optimal_offset = self.add_logged_quantity('optimal_offset', dtype=bool, si=False, ro=0,
                                                       initial=False)

        self.binning = self.add_logged_quantity('binning', dtype=int, ro=0,
                                                choices=[1, 2, 4], initial=1, reread_from_hardware_after_write=True)

        self.trsource = self.add_logged_quantity('trigger_source', dtype=str, si=False, ro=0,
                                                 choices=["internal", "external"], initial='internal',
                                                 reread_from_hardware_after_write=True)

        self.trmode = self.add_logged_quantity('trigger_mode', dtype=str, si=False, ro=0,
                                               choices=["normal", "start"], initial='normal',
                                               reread_from_hardware_after_write=True)

        self.trpolarity = self.add_logged_quantity('trigger_polarity', dtype=str, si=False, ro=0,
                                                   choices=["positive", "negative"], initial='positive',
                                                   reread_from_hardware_after_write=True)

        self.tractive = self.add_logged_quantity('trigger_active', dtype=str, si=False, ro=0,
                                                 choices=["edge", "level", "syncreadout"], initial='edge',
                                                 reread_from_hardware_after_write=True)

        self.ouchannel1 = self.add_logged_quantity('output_channel1', dtype=str, si=False, ro=0,
                                                   choices=["low", "exposure", "programmable", "triggerready", "high"],
                                                   initial='programmable', reread_from_hardware_after_write=True)

        self.ouchannel2 = self.add_logged_quantity('output_channel2', dtype=str, si=False, ro=0,
                                                   choices=["low", "exposure", "programmable", "triggerready", "high"],
                                                   initial='programmable', reread_from_hardware_after_write=True)

        self.ouchannel3 = self.add_logged_quantity('output_channel3', dtype=str, si=False, ro=0,
                                                   choices=["low", "exposure", "programmable", "triggerready", "high"],
                                                   initial='programmable', reread_from_hardware_after_write=True)
        self.outrsource1 = self.add_logged_quantity('output_trigger_source1', dtype=str, si=False, ro=0,
                                                    choices=["exposure", "readout_start", "readout_end",
                                                             "input_trigger_signal"],
                                                    initial='exposure', reread_from_hardware_after_write=True)
        self.outrsource2 = self.add_logged_quantity('output_trigger_source2', dtype=str, si=False, ro=0,
                                                    choices=["exposure", "readout_start", "readout_end",
                                                             "input_trigger_signal"],
                                                    initial='readout_start', reread_from_hardware_after_write=True)
        self.outrsource3 = self.add_logged_quantity('output_trigger_source3', dtype=str, si=False, ro=0,
                                                    choices=["exposure", "readout_start", "readout_end",
                                                             "input_trigger_signal"],
                                                    initial='readout_end', reread_from_hardware_after_write=True)

    #         self.preset_sizes = self.add_logged_quantity('preset_sizes', dtype=str, si=False, ro = 0,
    #                                                      choices = ["2048x2048",
    #                                                                 "2048x1024",
    #                                                                 '2048x512'
    #                                                                 '2048x256'
    #                                                                 '2048x'
    #                                                                 '2048x'
    #                                                                 '2048x'
    #                                                                 '2048x'
    #                                                                 '2048x'
    #                                                                 '2048x'
    #                                                                 '2048x'
    #                                                                 ''
    #                                                                 ''
    #                                                                 ''
    #                                                                 ''
    #                                                                 ''
    #                                                                 ''
    #                                                                 ])

    def connect(self):
        """
        The initial connection does not update the value in the device,
        since there is no set_from_hardware function, so the device has
        as initial values the values that we initialize in the HamamatsuDevice
        class. I'm struggling on how I can change this. There must be some function in
        ScopeFoundry
        """

        # self.trsource.change_readonly(True)
        # self.trmode.change_readonly(True)
        # self.trpolarity.change_readonly(True)
        # self.acquisition_mode.change_readonly(True) #if we change from run_till_abort to fixed_length while running it crashes

        self.hamamatsu = HamamatsuDevice(camera_id=0, frame_x=self.subarrayh.val, frame_y=self.subarrayv.val,
                                         acquisition_mode=self.acquisition_mode.val,
                                         number_frames=self.number_frames.val, exposure=self.exposure_time.val,
                                         trsource=self.trsource.val, trmode=self.trmode.val,
                                         trpolarity=self.trpolarity.val, tractive=self.tractive.val,
                                         ouchannel1=self.ouchannel1.val, ouchannel2=self.ouchannel2.val,
                                         ouchannel3=self.ouchannel3.val,
                                         # outrpolarity1=self.outrpolarity1.val,
                                         # outrpolarity2=self.outrpolarity2.val, outrpolarity3=self.outrpolarity3.val,
                                         outrsource1=self.outrsource1.val,
                                         outrsource2=self.outrsource2.val, outrsource3=self.outrsource3.val,
                                         subarrayh_pos=self.subarrayh_pos.val,
                                         subarrayv_pos=self.subarrayv_pos.val, binning=self.binning.val,
                                         hardware=self)  # maybe with more cameras we have to change

        self.readOnlyWhenOpt()
        self.camera.hardware_read_func = self.hamamatsu.getModelInfo
        self.temperature.hardware_read_func = self.hamamatsu.getTemperature
        self.submode.hardware_read_func = self.hamamatsu.setSubArrayMode
        self.exposure_time.hardware_read_func = self.hamamatsu.getExposure
        self.trsource.hardware_read_func = self.hamamatsu.getTriggerSource
        self.trmode.hardware_read_func = self.hamamatsu.getTriggerMode
        self.trpolarity.hardware_read_func = self.hamamatsu.getTriggerPolarity
        self.tractive.hardware_read_func = self.hamamatsu.getTriggerActive

        self.ouchannel1.hardware_read_func = self.hamamatsu.getOutputTrigger1
        self.ouchannel2.hardware_read_func = self.hamamatsu.getOutputTrigger2
        self.ouchannel3.hardware_read_func = self.hamamatsu.getOutputTrigger3
        # self.outrpolarity1.hardware_read_func = self.hamamatsu.getOutputTrigger1Polarity
        # self.outrpolarity2.hardware_read_func = self.hamamatsu.getOutputTrigger2Polarity
        # self.outrpolarity3.hardware_read_func = self.hamamatsu.getOutputTrigger3Polarity
        self.outrsource1.hardware__read_func = self.hamamatsu.getOutputTrigger1Source
        self.outrsource2.hardware__read_func = self.hamamatsu.getOutputTrigger2Source
        self.outrsource3.hardware__read_func = self.hamamatsu.getOutputTrigger3Source

        self.subarrayh.hardware_read_func = self.hamamatsu.getSubarrayH
        self.subarrayv.hardware_read_func = self.hamamatsu.getSubarrayV
        self.subarrayh_pos.hardware_read_func = self.hamamatsu.getSubarrayHpos
        self.subarrayv_pos.hardware_read_func = self.hamamatsu.getSubarrayVpos
        self.internal_frame_rate.hardware_read_func = self.hamamatsu.getInternalFrameRate
        self.binning.hardware_read_func = self.hamamatsu.getBinning

        self.subarrayh.hardware_set_func = self.hamamatsu.setSubarrayH
        self.subarrayv.hardware_set_func = self.hamamatsu.setSubarrayV
        self.subarrayh_pos.hardware_set_func = self.hamamatsu.setSubarrayHpos
        self.subarrayv_pos.hardware_set_func = self.hamamatsu.setSubarrayVpos
        self.exposure_time.hardware_set_func = self.hamamatsu.setExposure
        self.acquisition_mode.hardware_set_func = self.hamamatsu.setAcquisition
        self.number_frames.hardware_set_func = self.hamamatsu.setNumberImages
        self.trsource.hardware_set_func = self.hamamatsu.setTriggerSource
        self.trmode.hardware_set_func = self.hamamatsu.setTriggerMode
        self.trpolarity.hardware_set_func = self.hamamatsu.setTriggerPolarity
        self.tractive.hardware_set_func = self.hamamatsu.setTriggerActive

        self.ouchannel1.hardware_set_func = self.hamamatsu.setOutputTrigger1
        self.ouchannel2.hardware_set_func = self.hamamatsu.setOutputTrigger2
        self.ouchannel3.hardware_set_func = self.hamamatsu.setOutputTrigger3
        # self.outrpolarity1.hardware_set_func = self.hamamatsu.setOutputTrigger1Polarity
        # self.outrpolarity2.hardware_set_func = self.hamamatsu.setOutputTrigger2Polarity
        # self.outrpolarity3.hardware_set_func = self.hamamatsu.setOutputTrigger3Polarity
        self.outrsource1.hardware_set_func = self.hamamatsu.setOutputTrigger1Source
        self.outrsource2.hardware_set_func = self.hamamatsu.setOutputTrigger2Source
        self.outrsource3.hardware_set_func = self.hamamatsu.setOutputTrigger3Source

        self.binning.hardware_set_func = self.hamamatsu.setBinning
        self.optimal_offset.hardware_set_func = self.readOnlyWhenOpt

        self.read_from_hardware()  # read from hardware at connection

    #         self.subarrayh.update_value(2048)
    #         self.subarrayv.update_value(2048)
    #         self.exposure_time.update_value(0.01)
    #         self.acquisition_mode.update_value("fixed_length")
    #         self.number_frames.update_value(2)

    def disconnect(self):

        # self.trsource.change_readonly(False)
        # self.trmode.change_readonly(False)
        # self.trpolarity.change_readonly(False)

        if hasattr(self, 'hamamatsu'):
            self.hamamatsu.stopAcquisition()
            self.hamamatsu.shutdown()
            #             error_uninit = self.hamamatsu.dcam.dcamapi_uninit()
            #             if (error_uninit != DCAMERR_NOERROR):
            #                 raise DCAMException("DCAM uninitialization failed with error code " + str(error_uninit))
            del self.hamamatsu

        for lq in self.settings.as_list():
            lq.hardware_read_func = None
            lq.hardware_set_func = None

    def readOnlyWhenOpt(self, value=None):
        # Done for avoiding the changing of the subarray position in optimal offset mode
        # The "value" argument has no meaning, but I have to put at least one argument
        if self.optimal_offset.val:
            self.subarrayh_pos.change_readonly(True)
            self.subarrayv_pos.change_readonly(True)
            self.hamamatsu.setSubarrayHpos(self.hamamatsu.calculateOptimalPos(self.subarrayh.val))
            self.hamamatsu.setSubarrayVpos(self.hamamatsu.calculateOptimalPos(self.subarrayv.val))
        else:
            self.subarrayh_pos.change_readonly(False)
            self.subarrayv_pos.change_readonly(False)

    def updateCameraSettings(self):
        self.hamamatsu.frame_x = self.subarrayh.val
        self.hamamatsu.frame_y = self.subarrayv.val
        self.hamamatsu.acquisition_mode = self.acquisition_mode.val
        self.hamamatsu.number_frames = self.number_frames.val
        self.hamamatsu.exposure = self.exposure_time.val
        self.hamamatsu.trsource = self.trsource.val
        self.hamamatsu.trmode = self.trmode.val
        self.hamamatsu.trpolarity = self.trpolarity.val
        self.hamamatsu.tractive = self.tractive.val
        self.hamamatsu.ouchannel1 = self.ouchannel1.val
        self.hamamatsu.ouchannel2 = self.ouchannel2.val
        self.hamamatsu.ouchannel3 = self.ouchannel3.val
        # self.hamamatsu.outrpolarity = self.outrpolarity.val
        self.hamamatsu.outrsource1 = self.outrsource1.val
        self.hamamatsu.outrsource2 = self.outrsource2.val
        self.hamamatsu.outrsource3 = self.outrsource3.val
        self.hamamatsu.subarrayh_pos = self.subarrayh_pos.val
        self.hamamatsu.subarrayv_pos = self.subarrayv_pos.val
        self.hamamatsu.binning = self.binning.val

        # print('Camera setting updated')

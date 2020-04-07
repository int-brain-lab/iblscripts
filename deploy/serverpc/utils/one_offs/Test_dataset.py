# Mock dataset

fpga_trials = [
# Times in FPGA clock [s]
'ready_tone_in': gnagna, # array([ 117.47373456,  121.35742959])
'error_tone_in': gnagna, # array([ 117.80138383,  nan])
'valve_open': gnagna, #  array([nan,   130.47468086])
'stim_freeze': gnagna, # array([ 117.83882397,  122.4721859])
'stimOn_times': gnagna, # array([ 117.47287055,  121.35627758])
'stimOff_times': gnagna, # array([ 119.87729586,  124.49347372])
'iti_in': gnagna,   # array([ 119.80068756,  124.42243345])
'goCue_times': gnagna,  # array([ 117.47373456,  121.35742959])
'feedback_times': gnagna,   # array([ 117.80138383,  122.42341771])
'intervals': gnagna,    # array([[ 115.92419256,  119.80068756],  [5456.7324425 , 5463.76686972]])}
]
alf_trials = [
# Times in Bpod clock [s]
'goCueTrigger_times_bpod': gnagna,  # array([1.54880000e+00, 5.43220000e+00])
'response_times_bpod': gnagna,  # array([1.87650000e+00, 6.49820000e+00])
'intervals_bpod': gnagna,   # array([[0.0000000e+00, 4.3765030e+00], [5.3407663e+03, 5.3483007e+03]])
# Times from session start [s]
'goCueTrigger_times': gnagna,   # array([ 117.47299256,  121.35645899])
'response_times': gnagna,   # array([ 117.80069256,  122.42245899])
'intervals': gnagna,    # array([[ 115.92419256,  119.80068756], [5456.7324425 , 5463.76686972]])
'stimOn_times': gnagna, # array([ 117.47287055,  121.35627758])
'goCue_times': gnagna,  # array([ 117.47373456,  121.35742959])
'feedback_times': gnagna,   # array([ 117.80138383,  122.42341771])
]
ephysqc.qc_fpga_task(fpga_trials, None, alf_trials)
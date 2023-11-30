%% Setup Timeline
% This script sets up the Timeline object for a mesoscope rig. The
% channel configurations here are the same as the UCL mesoscope,
% however many channels can be digital instead of analogue, and may
% be any channel ID so long it correctly reflects the wiring.

% A complete guide to setting up and using Timeline can be found here:
% http://cortex-lab.github.io/Rigbox/Timeline.html

%% Creating a timeline object
timeline = hw.Timeline;

% Setting UseTimeline to true allows timeline to be started by default at
% the start of each experiment.  Otherwise it can be toggled on and off by
% pressing the 't' key while running SRV.EXPSERVER.
timeline.UseTimeline = true;

% The sample rate doesn't need to be too high for imaging.
timeline.DaqSampleRate = 2000;
timeline.DaqSamplesPerNotify = 2000;

% The DAQ device ID can be found in the NI device monitor.
timeline.DaqIds = 'Dev1';

% The stop delay determines how long to wait for the final samples to
% arrive from the DAQ before closing the device
timeline.StopDelay = 3; % seconds

% Expected imaging time (used for initializing data arrays)
timeline.MaxExpectedDuration = 14400;

% When true, it writes the data straight to a bin file allowing one to
% recover data if MATLAB crashes
timeline.WriteBufferToDisk = true;

%% Inputs
% This can be digital instead of analogue
timeline.Inputs.daqChannelID = 'ai0';
timeline.Inputs.measurement = 'Voltage';
timeline.Inputs.terminalConfig = 'SingleEnded';

%% Bpod inputs
% The Bpod BNC2 output
timeline.addInput('bpod', 'ai7', 'Voltage');
% The frame2ttl as Bpod BNC1 output. This can be a digital counter channel.
timeline.addInput('frame2ttl', 'ai11', 'Voltage');
% This measured the audio output TTLs from the Harp soundcard. This can be
% a digital counter channel.
timeline.addInput('audio', 'ai15', 'Voltage');
% The mouse wheel (see Kubler instructions on wiring X4).
% http://cortex-lab.github.io/Rigbox/Burgess_hardware_setup.html#h.sxj7qk4c4gl2
timeline.addInput('rotary_encoder', 'ctr3', 'Position', [], 10);

%% Camera inputs
% These could also be digital counter inputs
timeline.addInput('left_camera', 'ai12', 'Voltage');
timeline.addInput('right_camera', 'ai13', 'Voltage');
% The belly camera is not yet implemented but will replace the body camera
% used at ephys rigs.
timeline.addInput('belly_camera', 'ai14', 'Voltage');

%% Mesoscope inputs
% A pulse each time a new frame is acquired
timeline.addInput('neuralFrames', 'ctr0', 'EdgeCount');
% A pulse each time a new volume is acquired
timeline.addInput('volumeCounter', 'ctr2', 'EdgeCount');
% The galvometer X motor signal for tracking mirror movements
timeline.addInput('GalvoX', 'ai4', 'Voltage');
% The galvometer Y motor signal for tracking mirror movements
timeline.addInput('GalvoY', 'ai5', 'Voltage');
% Focus voltage signal
timeline.addInput('RemoteFocus1', 'ai8', 'Voltage');
timeline.addInput('RemoteFocus2', 'ai9', 'Voltage');
% Laser power signal (unclear how this maps to Watts, likely A.U.)
timeline.addInput('LaserPower', 'ai6', 'Voltage');
% Feedback from the aqcuire live output so we know exactly when it started.
% The acquire live signal triggers the ScanImage acquisition.
timeline.addInput('acqLive', 'ai2', 'Voltage');

%% Optional inputs
% In cortexlab we have a separate photodiode for Rigbox experiments.
% This could also be the analogue output signal of the frame2ttl
% photodiode.
timeline.addInput('photoDiode', 'ai0', 'Voltage');
% This is the raw valve voltage out when using an NResearch 225P011-21 PS
% valve with a feedback signal wire.
timeline.addInput('reward_valve', 'ai10', 'Voltage');
% This is the Bpod input but as a digital edge count. This is very useful
% as the sample rate is a lot higher.
timeline.addInput('bpod_rising_edge', 'ctr1', 'EdgeCount');

%% Set active inputs
% The UseInputs array contains list of channel names to save. Here we use
% all the inputs.
timeline.UseInputs = {timeline.Inputs.name};

%% Outputs
% They may be changed by setting the above fields, e.g.
timeline.Outputs(1).DaqDeviceID = timeline.DaqIds;
timeline.Outputs(1).DaqChannelID = 'port0/line0';

% The acquire live output is a single high voltage signal indicating when
% ScanImage is recording.
timeline.Outputs(2) = hw.TLOutputAcqLive;
timeline.Outputs(2).DaqDeviceID = timeline.DaqIds;
timeline.Outputs(2).DaqChannelID = 'port1/line2';

%% Wiring
% To set up chrono a wire must bridge the terminals defined in
% Outputs(1).DaqChannelID and Inputs(1).daqChannelID
% The current channal IDs are printed to the command by running the this:
timeline.wiringInfo('chrono');

%% Saving
% The hardware device settings are stored in a MAT file named 'hardware',
% defined in dat.paths
hardware = fullfile(getOr(dat.paths, 'rigConfig'), 'hardware.mat');
save(hardware, 'timeline')

%% Loading and modifying Timeline
load(hardware)
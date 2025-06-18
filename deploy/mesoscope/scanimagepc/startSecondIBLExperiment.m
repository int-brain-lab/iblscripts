function out = startSecondIBLExperiment(animalName)

% script to start an experiment within IBL environment


videoIP = '128.40.198.96'; % IBL-MESO-VIDEO
videoPort = 1001;
timelineIP = '128.40.198.195'; % ZIGZAG
timelinePort = 1001;
behavIP = '0.0.0.0'; % won't be used for now
behavPort = 1001;
mesoscopeIP = myIP; % That is this computer - we will be using the existing pipeline to start experiments
mesoscopePort = 1001;

%% generate an ExpRef for the new experiment

[ExpRef, expSeq] = dat.newExp(animalName);
[subject, iSeries, expNum] = dat.expRefToMpep(ExpRef);

%% generate a properly formatted ExpStart message

expStartMessage = sprintf('ExpStart %s %s %s', subject, num2str(iSeries), num2str(expNum));
blockStartMessage = sprintf('BlockStart %s %s %s', subject, num2str(iSeries), num2str(expNum));

%% Send UDPs to all the hosts in the correct order and correct timing
uMesoscope = udp(mesoscopeIP, mesoscopePort);
uTimeline = udp(timelineIP, timelinePort);
uVideo = udp(videoIP, videoPort);

fopen(uMesoscope);
fopen(uTimeline);
% fopen(uVideo);

fwrite(uMesoscope, expStartMessage);
fwrite(uTimeline, blockStartMessage);
% TODO confirm that timeline started
% monitor the mesoscope acquisition status and then trigger the cameras
% only after it started acquiring? (or actually wait for timeline UDP echo)
% and the easiest way would be to just wait a certain number of seconds (might be problematic, blocking)
% fwrite(uVideo, expStartMessage);

fclose(uMesoscope);
fclose(uTimeline);
% fclose(uVideo);

%% Create a listener that will send ExpEnd to Timeline (and video PC?) when acquisition stops

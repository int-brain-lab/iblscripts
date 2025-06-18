%% udp testing script

hSI.hScan2D.logFilePath = 'C:\Users\scanimage\Desktop\junk';
hSI.hScan2D.logFileStem = 'junk';
hSI.hScan2D.logFramesPerFile = 1000;
hSI.hScan2D.logAverageFactor = 1; %SP added on 16-06-2023
hSI.hChannels.loggingEnable = false;
hSI.hScan2D.trigAcqInTerm = 'D2.1';
hSI.hScan2D.trigAcqEdge = 'rising';
hSI.hScan2D.trigStopInTerm = 'D2.4';
hSI.hScan2D.trigStopEdge = 'rising';
hSI.hScan2D.trigNextStopEnable = false; % this line has no effect
hSI.hStackManager.framesPerSlice = Inf;

% rig=RigInfoGet;

% echoudp('on',1001)
u = udp('0.0.0.0', 9999, 'LocalPort', 1001);
u.UserData.hSI = hSI;
u.UserData.hSICtl = hSICtl;

% define DAQ session to be used for stopping acquisition
% u.UserData.daqSession = daq.createSession('ni');
% u.UserData.daqSession.addDigitalChannel('Dev1', 'port0/line0', 'OutputOnly');

%fprintf('Do you run IBL (1) or Rigbox(2)? ')
option = input('Do you run IBL (1) or Rigbox(2)? ')
switch option
    case 1
        fprintf('Loading IBL configuration of the listener\n');
        set(u, 'DatagramReceivedFcn', @SIMesoUDPCallback_IBL);
    case 2
        fprintf('Loading Rigbox configuration of the listener\n');
        set(u, 'DatagramReceivedFcn', @SIMesoUDPCallback_Rigbox);
    otherwise
        fprintf('Defaulting to IBL, ha-ha-ha\n');
        set(u, 'DatagramReceivedFcn', @SIMesoUDPCallback_IBL);
end
fopen(u);
%diskSpaceLeft('F:\');
% echoudp('off');

% fclose(u);
% delete(u);

% fclose(u); fopen(u);
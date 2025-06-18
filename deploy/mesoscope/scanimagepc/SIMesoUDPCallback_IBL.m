function SIMesoUDPCallback(src, evt)

% fprintf('%s:%d\n', u.DatagramAddress, u.DatagramPort),
% u.RemoteHost=u.DatagramAddress;
% u.RemotePort=u.DatagramPort;
% disp('now reading data');

% not really sure why this needs to be persistent, probably not
% persistent folders

stopDelay = 0; % delay in seconds between receiving ExpEnd and aborting (stopping) the acquisition
% The best practice is to use 0, and use the ExpEnd delay feature in mpep
% to acquire a few seconds of data after the last stimulus presentation.
% stopDelay in Timeline for ZCAMP3 is set to 1 second, and we definitely
% want to stop acquisition before stopping TL.

h = src.UserData.hSI;
hCtl = src.UserData.hSICtl;
% dS = src.UserData.daqSession;

ip=src.DatagramAddress;
port=src.DatagramPort;
useExtTrigger = true;
% sometimes it might be necessary to enable/disable hardware triggering
% depending on who is the master computer
% if isequal(ip, '128.40.198.70') % this is zurprize (srv/mc)
%     useExtTrigger = false;
% elseif isequal(ip, '128.40.198.72') % this is mpep on zimage
%     useExtTrigger = true;
% else
%     useExtTrigger = h.extTrigEnable;
% end
data=fread(src, src.BytesAvailable);
str=char(data');
fprintf('Received ''%s'' from %s:%d\n', str, ip, port);

info=dat.mpepMessageParse(str);

% update remote IP to that of the sender (port is standard mpep listening
% port as initialised in SIListener.m)
src.RemoteHost = ip;
src.RemotePort = port;

switch info.instruction
    case 'hello'
         fwrite(src, data);
    case 'ExpStart'
        % configure save filename and path
        rootFolder = dat.expPath(info.expRef, 'main', 'local');

        if ~exist(rootFolder)
            [mkSuccess, message] = mkdir(rootFolder);
            if mkSuccess
                fprintf('%s folder successfully created\n', rootFolder);
            else
                error('There was a problem creating %s. %s\n', rootFolder, message');
            end
        end
        
        % generating subfulder name for the next bout of ecquisitions
        pattern = 'raw_imaging_data_';
        folderList = dir(rootFolder);
        folderList = folderList([folderList.isdir]);
        folderList = folderList(startsWith({folderList.name}, pattern));
        if isempty(folderList)
            prepareExperiment(info.expRef, 'mesoscope', ...
                'computerID', 'ZcanImage', 'fullPathInSettings', true);
            acquisitionNumber = 0;
            dataFolder = fullfile(rootFolder, sprintf('%s%02d', pattern, acquisitionNumber));
        else
            % this code assumes a very specific naming, i.e. 'pattern' +
            % two digit integer
            nFolders = numel(folderList);
            nameParts = split({folderList.name}, pattern);
            suffixes = nameParts(end - nFolders + 1 : end);
            latestAcquisition = max(cellfun(@str2num, suffixes));
            acquisitionNumber = latestAcquisition + 1;
            dataFolder = fullfile(rootFolder, sprintf('%s%02d', pattern, acquisitionNumber));
        end
        if ~exist(dataFolder)
            [mkSuccess, message] = mkdir(dataFolder);
            if mkSuccess
                fprintf('%s folder successfully created\n', dataFolder);
            else
                error('There was a problem creating %s. %s\n', dataFolder, message');
            end
        end
        

        fileStem = [info.expRef, '_2P'];
        
        h.hScan2D.logFilePath = dataFolder;
        h.hScan2D.logFileStem = fileStem;
        h.hScan2D.logFileCounter = acquisitionNumber;
        h.hChannels.loggingEnable = 1;
        h.hStackManager.framesPerSlice = Inf;
        h.hScan2D.logAverageFactor = 1;


        if useExtTrigger
            h.extTrigEnable = 1;
        else
            h.extTrigEnable = 0;
        end
        h.startGrab;
        fwrite(src, data);
        
    case {'ExpEnd', 'ExpInterrupt'}
        % abort loop, if not aborted yet
        pause(stopDelay); % wait a bit before stopping imaging
        %click STOP ACQ to make sure the full cycle is done
        % we now receive a hardware TTL on D2.4 instead
        % send TTL to stop acquisition (after completing a volume)
%         fprintf('output 0\n');
%         dS.outputSingleScan(0); % make sure there will be a rising edge
%         pause(0.1);
%         fprintf('output 1\n');
%         dS.outputSingleScan(1);
%         fprintf('wait 5 seconds\n');
%         tWaitStart = tic;
%         while toc(tWaitStart)<5
%         pause(0.01); % wait enough time for the whole volume to finish before responding
%         end
%         fprintf('output 0\n');
%         dS.outputSingleScan(0); % make sure the signal is LOW

        
        % clicking ABORT will stop mid-cycle, suboptimal
%         pause(1);
%         abort(h);


        % wait until the acquisition actually stops
        %         for some reason this doesn't work
        tWaitStart = tic;
        while ~(isequal(h.acqState, 'idle'))
            pause(0.1);
%             if toc(tWaitStart)>3
%                 fprintf('Took too long to stop acquisition, aborting to allow UDP echo\n')
%                 abort(h); % something is wrong, click abort
%             end
        end
        
        fprintf('Acquisition stopped\n');
        
        h.hScan2D.logFilePath = 'C:\Users\scanimage\Desktop\junk';
        h.hScan2D.logFileStem = 'junk';
        h.hChannels.loggingEnable = 0;
        h.extTrigEnable = 0;
        
        fwrite(src, data);
        
        fprintf('Ready for new acquisition\n');
        
        
    case 'BlockStart'
        % TODO the indices are hardcoded, this need to change
%         startButton = hCtl.hManagedGUIs(28).Children(2).Children(1);
%         triggerControlsV5('pbAcqStart_Callback', startButton, [], guidata(startButton));
        hCtl.hModel.hScan2D.trigIssueSoftwareAcq();
        fwrite(src, data);
    case 'BlockEnd'
        fwrite(src, data);
    case 'StimStart'
        fwrite(src, data);
    case 'StimEnd'
        fwrite(src, data);
    case 'ExpComplete'
        rootFolder = dat.expPath(info.expRef, 'main', 'local');
        flagFileName = fullfile(rootFolder, 'transfer_me.flag');
        [fID, errmsg] = fopen(flagFileName, 'w');
        if ~isempty(errmsg)
            warning('Did not manage to create %s\n', flagFileName);
            warning('%s', errmsg)
        else
            fclose(fID);
        end
        fwrite(src, data);
    otherwise
        fprintf('Unknown instruction : %s', info.instruction);
        fwrite(src, data);
end


end
%===========================================================
%


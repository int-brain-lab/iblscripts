function [frameQC_frames,frameQC_names,badframes] = writeFrameQC(exptName, varargin)

% [FrameQC_frames, FrameQC_names, badframes] = writeFrameQC(exptName[,mode,QCframes,description])
%
% allows experimenter to log QC issues to frames of an individual scanimage
% acquisition, either automatically or manually, then saves these as arrays 
% in 'exptQC' and 'badframes' for IBL extraction pipeline.
% If 'mode' is set to 'auto', will run automatic QC detection as sepcified
% in runAutoFrameQC.
% If only an animal name is provided, will search for latest ExpRef.
% Run consecutively if there is more than one QC issue to log. For standard
% QC issues be sure to use the right keyword (e.g. 'PMT' and 'galvo')
%
% written by Samuel Picard (March 2023)

p = inputParser;
addRequired(p,'exptName',@ischar);
addOptional(p,'mode','manual',@(x) any(validatestring(x,{'manual','auto'})));
addOptional(p,'QCframes',[],@(x) ((isvector(x) || isscalar(x)) && (all(x) > 0) || isempty(x)));
addParameter(p,'description','',@ischar);
parse(p,exptName,varargin{:});

try 
    import ScanImageTiffReader.ScanImageTiffReader;
catch
    warning('no ScanImageTiffReader package found, will try to read header info without it.')
end

%% find data path of experiment (if only animal name provided, find latest ExpRef)
%try as a full data path
fileList = dir(fullfile(exptName,'*.tif'));
try
    if isempty(fileList)
        %then as animal name (latest ExpRef)
        if dat.subjectExists(exptName)
            ExpRefs = dat.listExps(exptName);
            ExpRef = ExpRefs{end};
            %then as ExpRef
        elseif ~contains(exptName,'\') && dat.expExists(exptName)
            ExpRef = exptName;
        else
            error('%s is not a valid animal name, expRef, or tiff data path.',exptName)
        end
        datPath = dat.expPath(ExpRef,'local');
        datPath = datPath{1};
    else
        datPath = fileList(1).folder;
    end
    fprintf('Found tiff data in %s\n',datPath);
catch
    error('Cannot parse the data path %s.',exptName)
end

%% load existing frameQC array, or make new array

QCfileDir = dir(fullfile(datPath, 'exptQC.mat'));
badframesFileDir = dir(fullfile(datPath, 'badframes.mat'));
if ~isempty(QCfileDir)
    fprintf('Loading existing exptQC file...\n');
    load(fullfile(datPath,QCfileDir.name));
    nFramesTot = length(frameQC_frames);
    if ~isempty(badframesFileDir)
        fprintf('Loading existing badframes file...\n');
        %badframes = readNPY(fullfile(datPath,badframesFileDir.name)); %this bugs for some reason...
        load(fullfile(datPath,badframesFileDir.name));
    else
        badframes = uint32([]);
    end
else
    fileList = dir(fullfile(datPath, '*.tif'));
    nFiles = numel(fileList);
    
    % %total nr of frames is highest final 'frameNumbers' value in the descriptions headers of all the tiffs
    % %THIS IS BUGGY IN SOME FOLDERS, UNSURE WHY
    %     lastFrameNrs = zeros(1,nFiles);
    %     lastAcqNrs = ones(1,nFiles);
    %     fprintf('Extracting metadata from tiff nr. ');
    %     for iFile = 1:nFiles
    %         %display a iFile/nFiles counter (and replace previous entry)
    %         if iFile>1
    %             for k=0:log10(iFile-1), fprintf('\b'); end
    %             for kk=0:log10(nFiles-1), fprintf('\b'); end
    %             fprintf('\b')
    %         end
    %         fprintf('%d/%d', iFile, nFiles);
    %         try
    %             headerInfo = ScanImageTiffReader(fullfile(fileList(iFile).folder, fileList(iFile).name)).descriptions();
    %             evalc(headerInfo{end});
    %         catch
    %             imgInfo = imfinfo(fullfile(fileList(iFile).folder, fileList(iFile).name));
    %             evalc(imgInfo(end).ImageDescription);
    %         end
    %         lastFrameNrs(iFile) = frameNumbers;
    %         lastAcqNrs(iFile) = acquisitionNumbers;
    %     end
    %     fprintf('\n')
    
    %total nr of frames is the final 'frameNumbers' value in the descriptions header of the last tiff
    try
        headerInfo = ScanImageTiffReader(fullfile(fileList(iFile).folder, fileList(iFile).name)).descriptions();
        evalc(headerInfo{end});
    catch
        imgInfo = imfinfo(fullfile(fileList(nFiles).folder, fileList(nFiles).name));
        evalc(imgInfo(end).ImageDescription);
    end
    lastFrameNrs = frameNumbers;
    lastAcqNrs = acquisitionNumbers;
    
    if unique(lastAcqNrs)==1
        nFramesTot = max(lastFrameNrs);
    else
        error('Unsure about total number of frames: there seems to be >1 acquisition for this ExpRef.')
    end
    
    if strcmp(p.Results.mode,'auto')
        fprintf('Running automatic QC on tiffs:\n')
        %options = {}; options.firstFrame = 1; options.lastFrame = Inf; options.frameStride = 24;
        [frameQC_frames,frameQC_names,badframes] = runAutoFrameQC(datPath); %,options);
        %fb = input('Press A to accept, O to overwrite, or RETURN to append with manual QC: ',"s");
        fb = 'a';
        if strcmpi(fb,'a')
            save(fullfile(datPath,'exptQC.mat'),'frameQC_frames','frameQC_names');
            fprintf('frameQC.mat updated!\n')
            save(fullfile(datPath,'badframes.mat'),'badframes');
            fprintf('badframes.mat updated!\n')
            return
        elseif strcmpi(fb,'o')
            frameQC_frames = uint8(zeros(1,nFramesTot));
            frameQC_names = {'ok'};
            badframes = uint32([]);
        else
            prompt0='y';
        end
    else
        frameQC_frames = uint8(zeros(1,nFramesTot));
        frameQC_names = {'ok'};
        badframes = uint32([]);
    end
end


%% append new frame QC info and save to disk

QCframes = [];%p.Results.QCframes;
description = 'ok';%p.Results.description;

%these are standard QC issues that should have same string ID across sessions
standardDescriptions = {'PMT off','galvos fault'};
standardKeywords = {'PMT','galvo'};
badframesFlag = 'N';

if isempty(p.Results.QCframes) && isempty(p.Results.description)
    %if ~exist('prompt0'), prompt0 = input("Any manual QC frames to report (y/n)? ", "s"); end
    %QCflag = strcmpi(prompt0,'Y');
    QCflag = false;
    if QCflag
        description = input("Brief description of QC issue: ", "s");
        prompt1 = sprintf('Issue starting at frame (out of total of %i): ',nFramesTot);
        prompt2 = sprintf('Issue finishing at frame (out of total of %i): ',nFramesTot);
        frame1 = input(prompt1);
        frame2 = input(prompt2);
        QCframes = [frame1:frame2];
    end
elseif isempty(p.Results.description)
    description = input("Brief description of QC issue: ", "s");
end
if QCflag
    if max(QCframes)>length(frameQC_frames)
        error('indices in frameQC exceed total nr. of frames in tiff stack')
    else
        %match QC description to standard descriptions if possible
        for iKeyword=1:length(standardKeywords)
            if contains(description,standardKeywords{iKeyword})
                description = standardDescriptions{iKeyword};
                fprintf('Issue logged: ''%s'' (frames [%i:%i] will also be excluded from suite2p)\n',description,QCframes(1),QCframes(end));
                badframesFlag = 'Y';
                break
            end
        end
        QCid = 1;
        while QCid <= length(frameQC_names)
            if strcmp(description,frameQC_names{QCid})
                break
            else
                QCid = QCid+1;
            end
        end
        
        %log into output variables
        frameQC_frames(QCframes) = uint8(QCid-1); %because zero-indexing
        frameQC_names{QCid} = description;
        if strcmpi(badframesFlag,'N')
            badframesFlag = input("Should these frames also be excluded from suite2p (y/n)? ","s");
        end
        if strcmpi(badframesFlag,'Y')
            badframes = uint32([badframes,QCframes-1]); %because of zero-indexing
        end
        
    end
end
save(fullfile(datPath,'exptQC.mat'),'frameQC_frames','frameQC_names');
fprintf('frameQC.mat saved!\n')
save(fullfile(datPath,'badframes.mat'),'badframes');
fprintf('badframes.mat saved!\n')

%writeNPY('badframes',fullfile(datPath,'badframes.npy'));
%fprintf('badframes.npy updated\n')


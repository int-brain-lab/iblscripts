function [frameQC_frames,frameQC_names,badframes] = writeFrameQC(exptName, varargin)

% [FrameQC_frames, FrameQC_names, badframes] = writeFrameQC(exptName[,mode,QCframes,description])
%
% allows experimenter to log QC issues to frames of an individual scanimage
% acquisition, either automatically or manually, then saves these as arrays
% in 'exptQC' and 'badframes' for IBL extraction pipeline.
% If 'mode' is set to 'auto', will run automatic QC detection as sepcified
% in runAutoFrameQC.
% If only an animal name is provided, will search for latest ExpRef.
% For standard QC issues be sure to use the right keyword (e.g. 'PMT' and 'galvo')
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
    fprintf('\nFound tiff data in %s\n',datPath);
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
    
    %check if we have more than one imaging bouts in the folder
    bout_nms = cellfun(@(x) str2num(x(end-13:end-10)),{fileList(:).name});
    [unique_bouts, iFirst] = unique(bout_nms);
    if length(unique_bouts)>1
        iLastFile = [iFirst(2:end)-1;nFiles];
    else
        iLastFile = nFiles;
    end        
    
    %total nr of frames is the sum of the 'frameNumbers' values in the
    %descriptions header of the last tiffs of each imaging bout
    nFramesTot = 0;
    for i = 1:length(unique_bouts)
        imgInfo = imfinfo(fullfile(fileList(iLastFile(i)).folder, fileList(iLastFile(i)).name));
        evalc(imgInfo(end).ImageDescription);
        nFramesTot = nFramesTot + frameNumbers;
    end
    
    if strcmp(p.Results.mode,'auto')
        fprintf('Running automatic QC on tiffs:\n')
        %options = {}; options.firstFrame = 1; options.lastFrame = Inf; options.frameStride = 24;
        [frameQC_frames,frameQC_names,badframes] = runAutoFrameQC(datPath); %,options);
        fb = input('Press A to accept, O to overwrite, or RETURN to append with manual QC: ',"s");
        %fb = 'a'; %UNCOMMENT THIS TO ACCEPT ALL BY DEFAULT
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
        fb = input('Press A to report no issues, or RETURN to list manual QC: ',"s");
        frameQC_frames = uint8(zeros(1,nFramesTot));
        frameQC_names = {'ok'};
        badframes = uint32([]);
        if strcmpi(fb,'a')
            prompt0='n';
        else
            prompt0='y';
        end
    end
end


%% append new frame QC info and save to disk

QCframes = [];%p.Results.QCframes;
description = 'ok';%p.Results.description;

%these are standard QC issues that should have same string ID across sessions
standardDescriptions = {'PMT off','galvos fault'};
standardKeywords = {'PMT','galvo'};
badframesFlag = 'N';

addManualQC_flag = false;

if isempty(p.Results.QCframes) && isempty(p.Results.description)
    addManualQC_flag = true;
    %prompt0 = 'y';
end

while addManualQC_flag
    
    if ~exist('prompt0'),
        prompt0 = input("Any manual QC frames to report (y/n)? ", "s");
    end
        
    QCflag = strcmpi(prompt0,'Y');
    %QCflag = false;
    if QCflag
        description = input("Brief description of QC issue: ", "s");
        prompt1 = sprintf('Issue starting at frame (out of total of %i): ',nFramesTot);
        prompt2 = sprintf('Issue finishing at frame (out of total of %i): ',nFramesTot);
        frame1 = input(prompt1);
        frame2 = input(prompt2);
        QCframes = [frame1:frame2];
    %else
    %    addManualQC_flag = false;
    %    prompt0 = 'n';
    end
    %elseif isempty(p.Results.description)
    %    description = input("Brief description of QC issue: ", "s");
    %end
    if QCflag
        if max(QCframes)>length(frameQC_frames)+1
            error('indices in frameQC exceed total nr. of frames in tiff stack')
        else
            %match QC description to standard descriptions if possible
            for iKeyword=1:length(standardKeywords)
                if contains(description,standardKeywords{iKeyword},'IgnoreCase',true)
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
    
    if ~strcmpi(prompt0,'n')
        prompt0 = input("Any additional manual QC frames to report (y/n)? ", "s");
    end
    if strcmpi(prompt0,'n')
        addManualQC_flag = false;
    end
end
save(fullfile(datPath,'exptQC.mat'),'frameQC_frames','frameQC_names');
fprintf('frameQC.mat saved!\n')
save(fullfile(datPath,'badframes.mat'),'badframes');
fprintf('badframes.mat saved!\n')

%writeNPY('badframes',fullfile(datPath,'badframes.npy'));
%fprintf('badframes.npy updated\n')


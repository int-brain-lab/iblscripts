function varargout = writeFrameQC(exptName, QCframes, description)

% [FrameQC_frames, FrameQC_name] = writeFrameQC(exptName, badframes, description)
%
% allows experimenter to log QC issues to frames of an individual scanimage
% acquisition, then saves these as arrays for IBL extraction pipeline.
% If only an animal name is provided, will search for latest ExpRef.
% Run consecutively if there is more than one QC issue to log. For standard
% QC issues be sure to use the right keyword (for now, 'PMT' and 'galvo')
%
% written by Samuel Picard (March 2023)

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
    
    frameQC_frames = uint8(zeros(1,nFramesTot));
    frameQC_name = {'ok'};
    badframes = uint32([]);
    
end

%% append new frame QC info and save to disk

%these are standard QC issues that should have same string ID across sessions
standardDescriptions = {'PMT off','galvos fault'};
standardKeywords = {'PMT','galvo'};
badframesFlag = 'N';

if nargin<2
    description = input("Brief description of QC issue: ", "s");
    prompt1 = sprintf('Issue starting at frame (out of total of %i): ',nFramesTot);
    prompt2 = sprintf('Issue finishing at frame (out of total of %i): ',nFramesTot);
    frame1 = input(prompt1);
    frame2 = input(prompt2);
    QCframes = [frame1:frame2];
elseif nargin<3
    description = input("Brief description of QC issue: ", "s");
end

if max(QCframes)>length(frameQC_frames)
    error('indices specified in frameQC exceed total nr. of frames in tiff stack')
else
    for iKeyword=1:length(standardKeywords)
        if contains(description,standardKeywords{iKeyword})
            description = standardDescriptions{iKeyword};
            fprintf('Issue logged: ''%s'' (frames [%i:%i] will also be excluded from suite2p)\n',description,QCframes(1),QCframes(end));
            badframesFlag = 'Y';
            break
        end
    end
    QCid = 1;
    while QCid <= length(frameQC_name)
        if strcmp(description,frameQC_name{QCid})
            break
        else
            QCid = QCid+1;
        end
    end
    frameQC_frames(QCframes) = uint8(QCid-1); %because zero-indexing
    frameQC_name{QCid} = description;
    if strcmpi(badframesFlag,'N')
        badframesFlag = input("Should these frames also be excluded from suite2p (Y/N)? ","s");
    end
    if strcmpi(badframesFlag,'Y')
        badframes = uint32([badframes,QCframes-1]); %because of zero-indexing
    end
end

save(fullfile(datPath,'exptQC.mat'),'frameQC_frames','frameQC_name');
fprintf('frameQC.mat updated!\n')
save(fullfile(datPath,'badframes.mat'),'badframes');
fprintf('badframes.mat updated!\n')

%writeNPY('badframes',fullfile(datPath,'badframes.npy'));
%fprintf('badframes.npy updated\n')


function meta = mesoscopeMetadataExtraction_ref(filename, varargin)

%meta = mesoscopeMetadataExtraction_ref(filename, varargin)
%
%similar to mesoscopeMetadataExtraction.m, but for structural reference
%stack. Produces a simpler meta-data file as well as a stitched tiff-stack for
%further registration.

if nargin < 1
    %filename = 'D:\Subjects\test\2023-01-31\1\2023-01-31_1_test_2P_00001_00001.tif';
    %filename = 'M:\Subjects\SP035\2023-03-03\001\raw_imaging_data_01\2023-03-03_1_SP035_2P_00001_00001.tif';
    error('Must provide a valid data path.')
end

% User parameters
p = inputParser;
p.addParameter('positiveML', [0, -1], @isnumeric)
p.addParameter('positiveAP', [-1, 0], @isnumeric)
p.addParameter('centerML', 2.7, @isnumeric)
p.addParameter('centerAP', -2.6, @isnumeric)
p.addParameter('alyx', Alyx('',''), @(v)isa(v,'Alyx'))
p.parse(varargin{:})

alyx = p.Results.alyx;

%% figure out file paths and parse animal name

%these are the regular expressions we expect
reg_expref = '(?<date>^[0-9\-]+)_(?<seq>\w+)_(?<subject>\w+)';
reg_tiff = [reg_expref '_(?<type>\w+)_(?<acq>\d+)_(?<file>\d+)'];

%try as full tiff file-path
if isfile(filename)
    [ff, fn, fext] = fileparts(filename);
    parsed = regexp(fn, reg_tiff, 'names');
    fn = [fn, fext];
    subj = parsed.subject;
    %subj = 'SP043';
    fileList = dir(fullfile(ff, fn)); %only take this tif
    %fileList = dir(fullfile(ff, ['*', fext])); %all tifs in folder
else
    %try as a final data path
    fileList = dir(fullfile(filename,'reference','*.tif'));
    try
        if isempty(fileList)
            %then as animal name (take latest ExpRef)
            if dat.subjectExists(filename)
                subj = filename;
                ExpRefs = dat.listExps(subj);
                ExpRef = ExpRefs{end};
                %then as ExpRef
            elseif ~contains(filename,'\') && dat.expExists(filename)
                ExpRef = filename;
            else
                error('%s is not a valid animal name, expRef, or tiff data path.',filename)
            end
            datPath = dat.expPath(ExpRef,'local');
            fileList = dir(fullfile(datPath{1},'reference','*.tif'));   
            if isempty(fileList)
                warning('%s does not exist or does not contain a tiff.\n',datPath{1});
            end
        end
        ff = fileList(1).folder;
        fn = fileList(1).name;
        parsed = regexp(fn, reg_tiff, 'names');
        subj = parsed.subject;
    catch
        error('Cannot parse the data path %s.',filename)
    end
end
fullfilepath = fullfile(ff,fn);
fprintf('%s\n',ff);

%% Generate the skeleton of the output struct
meta = struct();

% rig based
meta.channelID.green = [1, 2]; % information about channel numbers (red/green)
meta.channelID.red = [3, 4]; % information about channel numbers (red/green)
meta.channelID.primary = [2, 4]; % information about channel numbers (primary/secondary path)
meta.channelID.secondary = [1, 3]; % information about channel numbers (primary/secondary path)
meta.laserPowerCalibration.V = linspace(0, 5, 101); % calibration data imaging/dual etc.
meta.laserPowerCalibration.Prcnt = linspace(0, 100, 101); % calibration data imaging/dual etc.
meta.laserPowerCalibration.mW = linspace(0, 1200, 101); % calibration data imaging/dual etc.
meta.laserPowerCalibration.dualV = linspace(0, 2, 101); % calibration data imaging/dual etc.
meta.laserPowerCalibration.dualPrc = linspace(0, 100, 101); % calibration data imaging/dual etc.

% once per rig configuration (i.e. flipping axes, rotating animal)
meta.imageOrientation.positiveML = p.Results.positiveML;
meta.imageOrientation.positiveAP = p.Results.positiveAP;

% animal based
% extracted from ScanImage header
meta.centerDeg.x = NaN; % Centre of the reference circle in ScanImage coordinates
meta.centerDeg.y = NaN; % Centre of the ref circle - in ScanImage coordinates
meta.centerMM.x = NaN; % in mm, but still in SI coords
meta.centerMM.y = NaN; % in mm, but still in SI coords

% extracted from Alyx (if possible)
try
    [meta.centerMM.ML, meta.centerMM.AP] = getCraniotomyCoordinates(subj,'alyx',alyx); % from surgery - centre of the window
catch
    meta.centerMM.ML = p.Results.centerML;
    meta.centerMM.AP = p.Results.centerAP;
    warning('Could not find craniotomy coordinates in alyx, please upload using update_craniotomy.py... Writing dummy coordinates.');
    %TO DO input manually here? Abort script if not found?
end

% per single experiment
meta.rawScanImageMeta = struct; % SI config and all the header info from tiff
meta.PMTGain = []; %TO DO input manually
meta.channelSaved = [];


%%
% keyboard;
%%
fInfo = imfinfo(fullfilepath);

% these should be the same across all frames apart from timestamps and
% framenumbers in the ImageDescription field
meta.rawScanImageMeta.Artist = jsondecode(fInfo(1).Artist);
meta.rawScanImageMeta.ImageDescription = fInfo(1).ImageDescription;
meta.rawScanImageMeta.Software = fInfo(1).Software;
meta.rawScanImageMeta.Format = fInfo(1).Format;
meta.rawScanImageMeta.Width = fInfo(1).Width;
meta.rawScanImageMeta.Height = fInfo(1).Height;
meta.rawScanImageMeta.BitDepth = fInfo(1).BitDepth;
meta.rawScanImageMeta.ByteOrder = fInfo(1).ByteOrder;
meta.rawScanImageMeta.XResolution = fInfo(1).XResolution;
meta.rawScanImageMeta.YResolution = fInfo(1).YResolution;
meta.rawScanImageMeta.ResolutionUnit = fInfo(1).ResolutionUnit;

fArtist = meta.rawScanImageMeta.Artist;

fSoftware = splitlines(meta.rawScanImageMeta.Software);
% this will generate an SI structure, be careful not to overwrite things
for i = 1:length(fSoftware)
    evalc(fSoftware{i});
end

nFiles = numel(fileList);
nFramesAccum = 0;
fprintf('Extracting metadata from tiff nr. ');
for iFile = 1:nFiles
    
    %display a iFile/nFiles counter (and replace previous entry)
    if iFile>1
        for k=0:log10(iFile-1), fprintf('\b'); end
        for kk=0:log10(nFiles-1), fprintf('\b'); end
        fprintf('\b')
    end
    fprintf('%d/%d', iFile, nFiles);
    
    %read image data
    if iFile == 1
        rawData = readTiffFast(fullfilepath);
    else
        rawData = cat(3,rawData,readTiffFast(fullfilepath)); %this is a faster tiff loading function
    end
    
    fInfo = imfinfo(fullfile(fileList(iFile).folder, fileList(iFile).name));
    nFrames = numel(fInfo);
    for iFrame = 1:nFrames
        fImageDescription = splitlines(fInfo(iFrame).ImageDescription);
        fImageDescription = fImageDescription(1:end-1);
        for iLine = 1:numel(fImageDescription)
            str2eval = sprintf('imageDescription(%d).%s', iFrame + nFramesAccum, fImageDescription{iLine});
            evalc(str2eval);
        end
    end
    nFramesAccum = nFramesAccum + nFrames;
end
fprintf('\n')
meta.acquisitionStartTime = imageDescription(1).epoch;
meta.nFrames = nFramesAccum;
%TODO add nVolumeFrames (for multi-channel / multi-depth data)

%% useful SI parameters
meta.scanImageParams = struct('objectiveResolution', SI.objectiveResolution);
meta.scanImageParams.hScan2D = struct(...
    'flytoTimePerScanfield', SI.hScan2D.flytoTimePerScanfield,...
    'scannerFrequency', SI.hScan2D.scannerFrequency);
meta.scanImageParams.hFastZ = struct(...
    'enableFieldCurveCorr', SI.hFastZ.enableFieldCurveCorr,...
    'position', SI.hFastZ.position);
meta.scanImageParams.hRoiManager = struct(...
    'scanFramePeriod', SI.hRoiManager.scanFramePeriod,...
    'scanFrameRate', SI.hRoiManager.scanFrameRate,...
    'scanVolumeRate', SI.hRoiManager.scanVolumeRate,...
    'linePeriod', SI.hRoiManager.linePeriod);
meta.scanImageParams.hStackManager = struct(...
    'numSlices', SI.hStackManager.zs,...
    'zs', SI.hStackManager.zs,...
    'zsRelative', SI.hStackManager.zsRelative,...
    'zsAllActuators', SI.hStackManager.zsAllActuators);

meta.channelSaved = SI.hChannels.channelSave; %these were the channels being saved

%% Coordinate transformation from ScanImage to stereotactic coords

% center of the craniotomy in ScanImage coordinates (in mm)
% here assuming first coord is x and second is y, to be confirmed
try
    offset = SI.hDisplay.circleOffset;
catch
    offset = [0, 0]; %in old data, assume window was centered
end
meta.centerMM.x = offset(1)/1000; % defined in microns
meta.centerMM.y = offset(2)/1000;
meta.centerDeg.x = offset(1)/SI.objectiveResolution;
meta.centerDeg.y = offset(2)/SI.objectiveResolution;

% center of the craniotomy estimated during surgery (in stereotaxic coords)
centerML = meta.centerMM.ML;
centerAP = meta.centerMM.AP;

% Orientation of the image as it comes out of ScanImage
% Consider replacing this with {'up', 'down', 'left', 'right'}, but then it
% depends on the consistent coordinate system
% 'up' == [0, -1];
% 'down' == [0, 1];
% 'left' == [-1, 0];
% 'right' == [1, 0];
posML = meta.imageOrientation.positiveML; % this is pointing up in the image
posAP = meta.imageOrientation.positiveAP; % this is pointing left in the image

% rotation transformation is in the following form:
% [ML, AP] = [x - centerX, y - centerY, 1] * TF;

% defining conditions
% [posML, 1] * TF = [0, 1]*SI.objectiveResolution/1000 + [centerML, centerAP];
% [posAP, 1] * TF = [-1, 0]*SI.objectiveResolution/1000 + [centerML, centerAP];
% [0, 0, 1] * TF = [centerML, centerAP];

TF = pinv([posML, 1; posAP, 1; 0, 0, 1]) *...
    [[0, SI.objectiveResolution/1000; -SI.objectiveResolution/1000, 0; 0, 0] + repmat([centerML, centerAP], 3, 1)];
meta.coordsTF = TF;

%% produce stitched mean image stack
fprintf('Making stitched mean reference tiff fom raw data...\n ');
imgStack = meanImgFromSItiff_stack(fullfilepath);

%% save everything
jsonFileName = fullfile(ff, 'reference.meta.json');
% txt = jsonencode(meta, 'PrettyPrint', true);
txt = jsonencode(meta, 'ConvertInfAndNaN', false);
fid = fopen(jsonFileName, 'wt');
fwrite(fid, txt);
fclose(fid);

%matFileName = fullfile(ff, [fn, '.mat']);
%save(matFileName, 'meta');

end

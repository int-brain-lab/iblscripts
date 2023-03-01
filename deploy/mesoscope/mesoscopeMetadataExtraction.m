function meta = mesoscopeMetadataExtraction(filename, options)

if nargin < 1
    filename = 'D:\Subjects\test\2023-01-31\1\2023-01-31_1_test_2P_00001_00001.tif';
end
% TODO only default fill fields that do not exist in the options structure
if nargin < 2
    options = struct;
    % from surgery - centre of the window in mm in AP/ML coordinates
    options.centerMM.ML = 3;
    options.centerMM.AP = -5;
    % image orientations and seen in ScanImage window relative to AP/ML axes
    options.positiveML = [0, -1]; %'up'; [0,0] is top left corner, right and down are positive
    options.positiveAP = [-1, 0]; %'left';
end
[ff, fn, fext] = fileparts(filename);
% we will need the whole list of files to extract FPGA frametimes for each frame
% here assuming the files are sorted, and only relevant .tif files are
% present in the folder
% TODO to fool proof this with pattern matching
fileList = dir(fullfile(ff, ['*', fext]));

%% Generate the skeleton of the output struct
meta = struct();

% rig based
meta.channelID.green = [1, 2]; % information about channel numbers (red/green main/dual plain)
meta.channelID.red = [3, 4]; % information about channel numbers (red/green main/dual plain)
meta.laserPowerCalibration.V = linspace(0, 5, 101); % calibration data imaging/dual etc.
meta.laserPowerCalibration.Prcnt = linspace(0, 100, 101); % calibration data imaging/dual etc.
meta.laserPowerCalibration.mW = linspace(0, 1200, 101); % calibration data imaging/dual etc.
meta.laserPowerCalibration.dualV = linspace(0, 2, 101); % calibration data imaging/dual etc.
meta.laserPowerCalibration.dualPrc = linspace(0, 100, 101); % calibration data imaging/dual etc.
% once per rig configuration (i.e. flipping axes, rotating animal)
meta.imageOrientation.positiveML = options.positiveML;
meta.imageOrientation.positiveAP = options.positiveAP;

% animal based
% extracted from ScanImage header
meta.centerDeg.x = NaN; % Centre of the reference circle in ScanImage coordinates
meta.centerDeg.y = NaN; % Centre of the ref circle - in ScanImage coordinates
meta.centerMM.x = NaN; % in mm, but still in SI coords
meta.centerMM.y = NaN; % in mm, but still in SI coords
% the following should be extracted from Alyx
meta.centerMM.ML = options.centerMM.ML; % from surgery - centre of the window
meta.centerMM.AP = options.centerMM.AP; % from surgery -  - centre of the window

% per single experiment
meta.rawScanImageMeta = struct; % SI config and all the header info from tiff
meta.PMTGain = []; %

% for each FOV
% extracted directly from the header
% make sure dual plane case is taken care of
meta.FOV.topLeftDeg = [NaN, NaN, NaN];
meta.FOV.topRightDeg = [NaN, NaN, NaN];
meta.FOV.bottomLeftDeg = [NaN, NaN, NaN];
meta.FOV.bottomRightDeg = [NaN, NaN, NaN];
% after rotation we will get these:
meta.FOV.topLeftMM = [NaN, NaN, NaN];
meta.FOV.topRightMM = [NaN, NaN, NaN];
meta.FOV.bottomLeftMM = [NaN, NaN, NaN];
meta.FOV.bottomRightMM = [NaN, NaN, NaN];
% these are the valid lines that hold the data (sans black stripes)
meta.FOV.FPGATimestamps = []; % [nFrames, 1] - also save as a separate npy file
meta.FOV.lineIdx = NaN; % which lines belong to this FOV in a single tiff frame
meta.FOV.lineTimeShifts = NaN; % [nLines, nPixels] line acquisition time shift relative to the beginning of a tiff frame

meta.FOV.nXnYnZ = [NaN, NaN, 1]; % number of pixels in the images
% use meta.channelConfig to figure out correct indices
meta.FOV.channelIdx = []; % {[1, 3], [2, 4], [1, 2], [3, 4], [1,2,3,4]} - GreenChannels == 1,2, RedChannels = 3,4
meta.FOV.imagingLaserPowerPrc = [];
meta.FOV.dualPlaneLaserPowerPrc = [];
meta.FOV.laserPowerW = [];


%%
% keyboard;
%%
fInfo = imfinfo(filename);

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
for iFile = 1:nFiles
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
meta.acquisitionStartTime = imageDescription(1).epoch;

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
  'zs', SI.hStackManager.zs,...
  'zsRelative', SI.hStackManager.zsRelative,...
  'zsAllActuators', SI.hStackManager.zsAllActuators);

%% Coordinate transformation from ScanImage to stereotactic coords

% center of the craniotomy in ScanImage coordinates (in mm)
% here assuming first coord is x and second is y, to be confirmed
meta.centerMM.x = SI.hDisplay.circleOffset(1)/1000; % I think it is defined in microns
meta.centerMM.y = SI.hDisplay.circleOffset(2)/1000;
meta.centerDeg.x = SI.hDisplay.circleOffset(1)/SI.objectiveResolution;
meta.centerDeg.y = SI.hDisplay.circleOffset(2)/SI.objectiveResolution;

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

TF = pinv([posML, 1; posAP, 1; 0, 0, 1]) * [0, SI.objectiveResolution/1000; -SI.objectiveResolution/1000, 0; centerML, centerAP];
TF = pinv([posML, 1; posAP, 1; 0, 0, 1]) *...
    [[0, SI.objectiveResolution/1000; -SI.objectiveResolution/1000, 0; 0, 0] + repmat([centerML, centerAP], 3, 1)];
meta.coordsTF = TF;

%% process individual FOVs

nFOVs = numel(fArtist.RoiGroups.imagingRoiGroup.rois);
nLines = nan(nFOVs, 1);
for iFOV = 1:nFOVs
    cXY = fArtist.RoiGroups.imagingRoiGroup.rois(iFOV).scanfields.centerXY';
    sXY = fArtist.RoiGroups.imagingRoiGroup.rois(iFOV).scanfields.sizeXY';
    nXnY = fArtist.RoiGroups.imagingRoiGroup.rois(iFOV).scanfields(1).pixelResolutionXY';

    meta.FOV(iFOV).nXnYnZ = [nXnY, 1];
    meta.FOV(iFOV).topLeftDeg = cXY + sXY.*[-1, -1]/2;
    meta.FOV(iFOV).topRightDeg = cXY + sXY.*[1, -1]/2;
    meta.FOV(iFOV).bottomLeftDeg = cXY + sXY.*[-1, 1]/2;
    meta.FOV(iFOV).bottomRightDeg = cXY + sXY.*[1, 1]/2;

    centerDegXY = [meta.centerDeg.x, meta.centerDeg.y];
    meta.FOV(iFOV).topLeftMM = [meta.FOV(iFOV).topLeftDeg - centerDegXY, 1]*TF;
    meta.FOV(iFOV).topRightMM = [meta.FOV(iFOV).topRightDeg - centerDegXY, 1]*TF;
    meta.FOV(iFOV).bottomLeftMM = [meta.FOV(iFOV).bottomLeftDeg - centerDegXY, 1]*TF;
    meta.FOV(iFOV).bottomRightMM = [meta.FOV(iFOV).bottomRightDeg - centerDegXY, 1]*TF;

    nLines(iFOV) = meta.FOV(iFOV).nXnYnZ(2);
end

nValidLines = sum(nLines);
nLinesPerGap = (fInfo(1).Height - nValidLines) / (nFOVs - 1);
fovStartIdx = [1; cumsum(nLines(1:end-1) + nLinesPerGap) + 1];
fovEndIdx = fovStartIdx + nLines - 1;

% Save timestamps per FOV
for iFOV = 1:nFOVs
    meta.FOV(iFOV).lineIdx = [fovStartIdx(iFOV):fovEndIdx(iFOV)]';
    fovTimeShift = (fovStartIdx(iFOV) - 1)*SI.hRoiManager.linePeriod;
    meta.FOV(iFOV).FPGATimestamps = [imageDescription.frameTimestamps_sec]' + fovTimeShift; 
    meta.FOV(iFOV).lineTimeShifts = [0:nLines(iFOV)-1]'*SI.hRoiManager.linePeriod;
    %     meta.FOV(iFOV).timeShifts = fovStartIdx(iFOV)*SI.hRoiManager.linePeriod;
end

% Save raw FPGA timestamps array
timestamps_filename = fullfile(ff, 'rawImagingData.times_scanImage.npy');
writeNPY([imageDescription.frameTimestamps_sec]', timestamps_filename)

jsonFileName = fullfile(ff, '_ibl_rawImagingData.meta.json');
% txt = jsonencode(meta, 'PrettyPrint', true);
txt = jsonencode(meta, 'ConvertInfAndNaN', false);
fid = fopen(jsonFileName, 'wt');
fwrite(fid, txt);
fclose(fid);

matFileName = fullfile(ff, [fn, '.mat']);
save(matFileName, 'meta');
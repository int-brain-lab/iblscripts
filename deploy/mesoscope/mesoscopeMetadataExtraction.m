function meta = mesoscopeMetadataExtraction(filename, varargin)

if nargin < 1
    %filename = 'D:\Subjects\test\2023-01-31\1\2023-01-31_1_test_2P_00001_00001.tif';
    %filename = 'M:\Subjects\SP035\2023-03-03\001\raw_imaging_data_01\2023-03-03_1_SP035_2P_00001_00001.tif';
    error('Must provide a valid data path.')
end

% User parameters
p = inputParser;
p.addParameter('positiveML', [0, -1], @isnumeric) %ML goes from medial to lateral. Opposite of Y-galvo axis.
p.addParameter('positiveAP', [-1, 0], @isnumeric) %AP goes from posterior to anterior. Opposite of X-galvo axis.
p.addParameter('centerML', 2.6, @isnumeric)
p.addParameter('centerAP', -2.0, @isnumeric)
p.addParameter('alyx', Alyx('',''), @(v)isa(v,'Alyx'))
p.parse(varargin{:})

alyx = p.Results.alyx;

%% figure out file paths and parse animal name

%these are the regular expressions we expect
reg_expref = '(?<date>^[0-9\-]+)_(?<seq>\w+)_(?<subject>\w+)';
reg_tiff = [reg_expref '_(?<type>\w+)_(?<acq>\d+)_(?<file>\d+)'];

%try as full tiff (or tif.mat) file-path
if isfile(filename)
    [ff, fn, fext] = fileparts(filename);
    parsed = regexp(fn, reg_tiff, 'names');
    fn = [fn, fext];
    subj = parsed.subject;
    % we will need the whole list of files to extract FPGA frametimes for each frame
    % here assuming the files are sorted, and only relevant .tif files are
    % present in the folder
    % TODO to fool proof this with pattern matching
    fileList = dir(fullfile(ff, ['*', fext]));
else
    %try as a final data path (first look for existing metadata structure, then try raw tifs)
    fileList = dir(fullfile(filename,'*2P*.mat'));
    try
        if isempty(fileList)
            fileList = dir(fullfile(filename,'*.tif'));
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
                fileList = dir(fullfile(datPath{1},'*.tif'));
                if isempty(fileList)
                    warning('%s does not exist or does not contain tiffs\n',datPath{1});
                end
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

[ff, fn, fext] = fileparts(fullfilepath);
if strcmp(fext,'.mat')
    load(fullfilepath); %this loads the already computed meta struct and uses the rawSIMeta field
    fprintf('Starting from previously extracted rawScanImageMeta, re-computing meta-data...\n');
    meta_exists = true;
else
    meta_exists = false;
    meta = struct;
end

meta.version = '0.2.0';

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
    warning('Could not find craniotomy coordinates in alyx, please upload using update_craniotomy.py... Using default coordinates!');
    %TO DO input manually here? Abort script if not found?
end
sprintf('Using the following coordinate: [%.1f %.1f]', meta.centerMM.ML, meta.centerMM.AP);

% per single experiment
if ~meta_exists
    meta.rawScanImageMeta = struct; % SI config and all the header info from tiff
end
meta.PMTGain = []; %TO DO input manually
meta.channelSaved = [];

% for each FOV (extracted directly from the header)
meta.FOV = struct;
% TODO make sure dual plane case is taken care of
meta.FOV.slice_id = NaN; %unique identifier of this FOV's slice in stack
meta.FOV.roiUUID = NaN; %unique identifier of this FOV's roi
meta.FOV.Zs = NaN; % RF depth(s) of the slice (length 2 for dual-plane?)
% coordinates of all corners of the FOV in scope coords
meta.FOV.Deg.topLeft = [NaN, NaN, NaN];
meta.FOV.Deg.topRight = [NaN, NaN, NaN];
meta.FOV.Deg.bottomLeft = [NaN, NaN, NaN];
meta.FOV.Deg.bottomRight = [NaN, NaN, NaN];
% coordinates after 3D projection
meta.FOV.MM.topLeft = [NaN, NaN, NaN];
meta.FOV.MM.topRight = [NaN, NaN, NaN];
meta.FOV.MM.bottomLeft = [NaN, NaN, NaN];
meta.FOV.MM.bottomRight = [NaN, NaN, NaN];

% these are the valid lines that hold the data (sans black stripes)
meta.FOV.lineIdx = NaN; % which lines belong to this FOV in a single tiff frame
%meta.FOV.lineTimeShifts = NaN; % [nLines, nPixels] line acquisition time shift relative to the beginning of a tiff frame

meta.FOV.nXnYnZ = [NaN, NaN, 1]; % number of pixels in the images
% use meta.channelConfig to figure out correct indices
% meta.FOV.channelIdx = []; % {[1, 3], [2, 4], [1, 2], [3, 4], [1,2,3,4]} - GreenChannels == 1,2, RedChannels = 3,4
%meta.FOV.imagingLaserPowerPrc = [];
%meta.FOV.dualPlaneLaserPowerPrc = [];
%meta.FOV.laserPowerW = [];

%%
% keyboard;
%% read raw metadata
%if we did not already do this, extract metadata from the tiff header.
if ~meta_exists
    
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
    
    nFiles = numel(fileList);
    %nFiles = 1; %for debugging
    nFramesAccum = 0;
    fprintf('Extracting metadata from tiff nr. ');
    for iFile = 1:nFiles
        
        %display a iFile/nFiles counter (and replace previous entry)
        if iFile>1
            for k=0:log10(iFile-1), fprintf('\b'); end
            for kk=0:log10(nFiles), fprintf('\b'); end
            fprintf('\b')
        end
        fprintf('%d/%d', iFile, nFiles);
        
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
    
    % Save raw FPGA timestamps array
    timestamps_filename = fullfile(ff, 'rawImagingData.times_scanImage.npy');
    writeNPY([imageDescription.frameTimestamps_sec]', timestamps_filename)
    
end

fArtist = meta.rawScanImageMeta.Artist;

fSoftware = splitlines(meta.rawScanImageMeta.Software);
% this will generate an SI structure, be careful not to overwrite things
for i = 1:length(fSoftware)
    evalc(fSoftware{i});
end


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
    [[SI.objectiveResolution/1000, 0; 0, SI.objectiveResolution/1000; 0, 0] + repmat([centerML, centerAP], 3, 1)];
meta.coordsTF = round(TF,3);

%% process individual FOVs
si_rois_all = fArtist.RoiGroups.imagingRoiGroup.rois;
si_rois = si_rois_all(logical([si_rois_all.enable])); %only consider the rois that were 'enabled'

%we define a IBL FOV as a unique combination of 'scanimage roi' and 'slice depth'

%look up the depths we used
% if SI.hStackManager.enable
Zs = SI.hStackManager.zs;
% else
%     Zs = [si_rois.zs];
% end
Zvals = unique(Zs); %NB this automatically sorts ascendingly
nZs = length(Zvals); %total number of depths defined
nrois = numel(si_rois); %total number of si_rois defined

%create a logical array of nrois * ndepths telling us at which depth a FOV exists
i_roi_depth = false(nrois,nZs);
for iroi = 1:nrois
    if si_rois(iroi).discretePlaneMode
        for iSlice = 1:length(si_rois(iroi).zs)
            [~,~,ii] = intersect(si_rois(iroi).zs(iSlice),Zvals);
            i_roi_depth(iroi,ii) = true;
        end
    else
        i_roi_depth(iroi,:) = true;
    end
end
nFOVs = sum(i_roi_depth(:));

%get info for each FOV
iFOV = 0;
for iSlice = 1:nZs
    for iRoi = 1:nrois
        if i_roi_depth(iRoi,iSlice)
            
            iFOV = iFOV+1;
            
            cXY = si_rois(iRoi).scanfields(1).centerXY';
            sXY = si_rois(iRoi).scanfields(1).sizeXY';
            nXnY = si_rois(iRoi).scanfields(1).pixelResolutionXY';
            
            meta.FOV(iFOV).slice_id = iSlice-1; %assuming 0-indexing
            meta.FOV(iFOV).roiUUID = si_rois(iRoi).roiUuid; %this is scanimage ID
            meta.FOV(iFOV).Zs = Zvals(iSlice);
            
            meta.FOV(iFOV).nXnYnZ = [nXnY, 1];
            meta.FOV(iFOV).Deg.topLeft = cXY + sXY.*[-1, -1]/2;
            meta.FOV(iFOV).Deg.topRight = cXY + sXY.*[1, -1]/2;
            meta.FOV(iFOV).Deg.bottomLeft = cXY + sXY.*[-1, 1]/2;
            meta.FOV(iFOV).Deg.bottomRight = cXY + sXY.*[1, 1]/2;
            
            centerDegXY = [meta.centerDeg.x, meta.centerDeg.y];
            meta.FOV(iFOV).MM.topLeft = [meta.FOV(iFOV).Deg.topLeft - centerDegXY, 1]*TF;
            meta.FOV(iFOV).MM.topRight = [meta.FOV(iFOV).Deg.topRight - centerDegXY, 1]*TF;
            meta.FOV(iFOV).MM.bottomLeft = [meta.FOV(iFOV).Deg.bottomLeft - centerDegXY, 1]*TF;
            meta.FOV(iFOV).MM.bottomRight = [meta.FOV(iFOV).Deg.bottomRight - centerDegXY, 1]*TF;
            
            nLines_allFOVs(iFOV) = meta.FOV(iFOV).nXnYnZ(2);
        end
    end
end

%maxnFOVsPerZ = ceil(nFOVs/nZs); %might be useful if nFOVs per Z isn't equal across Zs

%find which lines correspond to which FOV in each slice in the z-stack
%WARNING: this works for adjacent FOVs with the same nr of lines, but hasn't been tested yet for multi-plane data with unequal FOV sizes
nLines = cell(1, nZs);
nValidLines = zeros(1, nZs);
for iSlice = 1:nZs
    % get FOV info for each slice in the z-stack
    iFOVs_at_this_z = [meta.FOV.Zs]==Zvals(iSlice);
    nFOVs_at_this_z(iSlice) = sum(iFOVs_at_this_z);
    nValidLines(iSlice) = sum(nLines_allFOVs(iFOVs_at_this_z));
    nLines{iSlice} = nLines_allFOVs(iFOVs_at_this_z);
    nLinesPerGap = (meta.rawScanImageMeta.Height - nValidLines(iSlice)) / (nFOVs_at_this_z(iSlice) - 1);
    fovStartIdx = [1, cumsum(nLines{iSlice}(1:end-1) + nLinesPerGap) + 1];
    fovEndIdx = fovStartIdx + nLines{iSlice} - 1;
    iFOVs = find(iFOVs_at_this_z);
    % Save line indexes per FOV per z-slice
    for ii = 1:nFOVs_at_this_z(iSlice)
        iFOV = iFOVs(ii);
        meta.FOV(iFOV).lineIdx = (fovStartIdx(ii):fovEndIdx(ii))';
    end
end
%TODO: figure out how to work with 'volume frames' for multi-plane data!

% [ff_top,ff_raw] = fileparts(ff);

% 3D projection on the brain surface
%TODO: only need to do this for the first raw_imaging_data folder
% meta = projectMLAPDV(meta);
%
% % Save 3D projection data in separate npy files
% % saves them as 'stitched arrays', similar to the raw tiff frames where the
% % different FOVs are concatenated vertically
% for iFOV = 1:length(meta.FOV)
%     ff_alf = fullfile(ff_top,'alf',['FOV_0' num2str(iFOV)-1]);
%     if ~exist(ff_alf,'dir')
%         mkdir(ff_alf);
%         mlapdv = meta.FOV(iFOV).pixelMLAPDV * 1e3;  % Save in um instead of mm
%         mlapdv_filename = fullfile(ff_alf,'mpciMeanImage.mlapdv_estimate.npy');
%         atlasAnnotation = meta.FOV(iFOV).pixelAnnot;
%         annotation_filename = fullfile(ff_alf, 'mpciMeanImage.brainLocationIds_ccf_2017_estimate.npy');
%         writeNPY(mlapdv, mlapdv_filename)
%         writeNPY(atlasAnnotation, annotation_filename)
%     end
% end
% % remove the big data arrays from json
% metaForJson = meta;
% FOV = metaForJson.FOV;
% %for iFOV = 1:numel(metaForJson.FOV)
% FOV = rmfield(FOV,{'pixelMLAPDV','pixelAnnot'});
% %end
% metaForJson.FOV = FOV;

%% save everything
jsonFileName = fullfile(ff, '_ibl_rawImagingData.meta.json');
% txt = jsonencode(meta, 'PrettyPrint', true);
txt = jsonencode(meta, 'ConvertInfAndNaN', false);
fid = fopen(jsonFileName, 'wt');
fwrite(fid, txt);
fclose(fid);

matFileName = fullfile(ff, [fn, '.mat']);
save(matFileName, 'meta');

fprintf('Done!\n')

end

%%============================================================================================
% functions

function metaOut = projectMLAPDV(meta)

metaOut = meta;
% prepare for 3D projection onto brain surface
% TODO: get these atlas files onto a remote repo?
try
    surfaceData = load('mlapdvAtlas.mat');
catch
    surfaceData = load('C:\Users\scanimage\Documents\MATLAB\mlapdvAtlas.mat');
end
annotationsFile = 'annotation_volume_10um_by_index.npy';
try
    allenAV = readNPY(annotationsFile);
catch
    dataFolder = 'C:\Users\scanimage\Documents\allenCCFData';
    allenAV = readNPY(fullfile(dataFolder, annotationsFile));
end

coordML = meta.centerMM.ML;
coordAP = meta.centerMM.AP;
faceInd = surfaceData.flatTR.pointLocation(coordML, coordAP);
normalVector = surfaceData.dorsalTR.faceNormal(faceInd);

% find the coordDV that sits on the triangular face and had [coordML, coordAP] coordinates
% the three vertices defining the triangle
faceVertices = surfaceData.dorsalTR.Points(surfaceData.dorsalTR.ConnectivityList(faceInd, :), :);
% all the vertices should be on the plane ax + by + cz = 1, so we can find
% the abc coeffcicents by inverting the three equations for the three
% vertices
abc = faceVertices\[1;1;1];
% and then find a point on that plane that corresponds to a given x-y
% coordinate (which is ML-AP corrdinate)
coordDV = (1 - [coordML, coordAP]*[abc(1); abc(2)])/abc(3);
% We should not use the actual surface of the brain for this, as it might
% be in one of the sulci/valleys
% DO NOT USE THIS:
% coordDV = interp2(axisMLmm, axisAPmm, surfaceDV, coordML, coordAP);

% Now we need to span the plane of the coverslip with two orthogonal unit vectors
% We start with vY, because the order is important and we usually have less
% tilt along AP (pitch), wich will cause less deviation in vX from pure ML
vY = [0, normalVector(3), -normalVector(2)]; % orthogonal to the normal of the plane
vX = cross(vY, normalVector); % orthogonal to n and to vY
% normalize and flip the sign if necessary
vX = vX/norm(vX)*sign(vX(1));
vY = vY/norm(vY)*sign(vY(2));

% projection of FOVs on the brain surface to get ML-AP-DV coordinates
fprintf('Projecting in 3D:\n')
nFOVs = numel(meta.FOV);
for iFOV = 1:nFOVs
    fovStartTime = tic;
    fprintf('FOV %d/%d..', iFOV, nFOVs);
    [xPixIdx, yPixIdx] = meshgrid(1:meta.FOV(iFOV).nXnYnZ(1), 1:meta.FOV(iFOV).nXnYnZ(2));
    % xx and yy are in mm in coverslip space
    xx = interp2([1, meta.FOV(iFOV).nXnYnZ(1)], [1, meta.FOV(iFOV).nXnYnZ(2)], ...
        [meta.FOV(iFOV).MM.topLeft(1), meta.FOV(iFOV).MM.topRight(1); ...
        meta.FOV(iFOV).MM.bottomLeft(1), meta.FOV(iFOV).MM.bottomRight(1)], xPixIdx, yPixIdx);
    yy = interp2([1, meta.FOV(iFOV).nXnYnZ(1)], [1, meta.FOV(iFOV).nXnYnZ(2)], ...
        [meta.FOV(iFOV).MM.topLeft(2), meta.FOV(iFOV).MM.topRight(2); ...
        meta.FOV(iFOV).MM.bottomLeft(2), meta.FOV(iFOV).MM.bottomRight(2)], xPixIdx, yPixIdx);
    xx = xx(:) - coordML;
    yy = yy(:) - coordAP;
    
    % rotate xx and yy in 3D
    % coords they are still on the coverslip, but now have 3D values
    coords = (bsxfun(@times, vX, xx) + bsxfun(@times, vY, yy));
    coords = bsxfun(@plus, coords, [coordML, coordAP, coordDV]);
    
    % for each point of the FOV create a line parametrization
    % (trajectory normal to the coverslip plane)
    t = [-surfaceData.voxelSize:surfaceData.voxelSize:3]'; % start just above the coverslip and go 3 mm down, should be enough to 'meet' the brain
    % passing through the center of the craniotomy/coverslip
    trajCoordsCentered = bsxfun(@times, -normalVector, t);
    MLAPDV = nan(size(coords));
    annotation = nan(size(coords, 1), 1);
    nPoints = size(coords, 1);
    trajCoordsIdx = cell(nPoints, 1);
    parfor iPoint = 1:nPoints
        % shifted to the correct point on the coverslip, in true ML-AP-DV coords
        trajCoords = bsxfun(@plus, trajCoordsCentered, coords(iPoint, :));
        
        % find intersection coordinate with the brain
        
        % only use coordinates that exist in the atlas (kind of nearest neighbor
        % interpolation)
        trajCoordsNearest = [interp1(surfaceData.axisMLmm, surfaceData.axisMLmm, trajCoords(:,1), 'nearest', 'extrap'), ...
            interp1(surfaceData.axisAPmm, surfaceData.axisAPmm, trajCoords(:,2), 'nearest', 'extrap'), ...
            interp1(surfaceData.axisDVmm, surfaceData.axisDVmm, trajCoords(:,3), 'nearest', 'extrap')];
        
        % find indices from mm
        %         clear trajCoordsIdx;
        [~, trajCoordsIdx{iPoint}(:,1)] = ismember(trajCoordsNearest(:, 1), surfaceData.axisMLmm);
        [~, trajCoordsIdx{iPoint}(:,2)] = ismember(trajCoordsNearest(:, 2), surfaceData.axisAPmm);
        [~, trajCoordsIdx{iPoint}(:,3)] = ismember(trajCoordsNearest(:, 3), surfaceData.axisDVmm);
        
        anno = allenAV(sub2ind(size(allenAV), trajCoordsIdx{iPoint}(:, 2), trajCoordsIdx{iPoint}(:, 3), trajCoordsIdx{iPoint}(:, 1)));
        ind = find(anno~=1, 1, 'first');
        area = anno(ind);
        point = trajCoords(ind, :);
        if ~isempty(point)
            MLAPDV(iPoint, :) = point;
        else
            MLAPDV(iPoint, :) = nan(1, 3);
        end
        if ~isempty(ind)
            annotation(iPoint) = anno(ind);
        else
            annotation(iPoint) = NaN;
        end
        
    end
    
    fprintf('.done (%3.1f seconds)\n', toc(fovStartTime));
    metaOut.FOV(iFOV).pixelMLAPDV = reshape(MLAPDV, [size(xPixIdx), 3]);
    metaOut.FOV(iFOV).pixelAnnot = reshape(annotation, size(xPixIdx));
    
    % check dimensions (i.e. do we need a row or a column vector?)
    metaOut.FOV(iFOV).MLAPDV.topLeft = squeeze(metaOut.FOV(iFOV).pixelMLAPDV(1, 1, :));
    metaOut.FOV(iFOV).MLAPDV.topRight = squeeze(metaOut.FOV(iFOV).pixelMLAPDV(1, end, :));
    metaOut.FOV(iFOV).MLAPDV.bottomLeft = squeeze(metaOut.FOV(iFOV).pixelMLAPDV(end, 1, :));
    metaOut.FOV(iFOV).MLAPDV.bottomRight = squeeze(metaOut.FOV(iFOV).pixelMLAPDV(end, end, :));
    metaOut.FOV(iFOV).MLAPDV.center = squeeze(metaOut.FOV(iFOV).pixelMLAPDV(round(end/2), round(end/2), :));
    
    % We probably also want to save the brain regions of the
    % corners/centers of FOV (annotation field)
    metaOut.FOV(iFOV).brainLocationIds.topLeft = squeeze(metaOut.FOV(iFOV).pixelAnnot(1, 1));
    metaOut.FOV(iFOV).brainLocationIds.topRight = squeeze(metaOut.FOV(iFOV).pixelAnnot(1, end));
    metaOut.FOV(iFOV).brainLocationIds.bottomLeft = squeeze(metaOut.FOV(iFOV).pixelAnnot(end, 1));
    metaOut.FOV(iFOV).brainLocationIds.bottomRight = squeeze(metaOut.FOV(iFOV).pixelAnnot(end, end));
    metaOut.FOV(iFOV).brainLocationIds.center = squeeze(metaOut.FOV(iFOV).pixelAnnot(round(end/2), round(end/2)));
    
end

end
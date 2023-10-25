function imgStack = meanImgFromSItiff_stack(filename,options)

%makes a stitched mean image from a scanimage tiff containing one or more FOVs
%requirements: readTiffFast.m, loadFramesBuff.m, nFrames.m and nFramesTiff.m (from tiffTools)
%written by M Krumin, edited by Samuel Picard (Oct 2022)
%Feb 2023: SP included functionality for FOVs of different depths (discrete plane mode)
%Mar 2023: SP now only plotting considering scanimage ROIs that were 'enabled'
%Jul 2023: SP making tiff stack of stitched grid
%
% WARNING: this assumes FOVs are adjacent 'strips' of roughly equal pixel resolution
% and that each FOV was acquired vertically across the full stack

if nargin<2 || isempty(options)
    options.firstFrame = 1;
    options.lastFrame = Inf;
    options.frameStride = 1; % useful for reading only a specific channel/plane
end

[ff, fn, fext] = fileparts(filename);
savefolder = ff;
savename = 'referenceImage.stack.tif';

files_to_delete = {savename, 'reference.stack.tif', 'reference.meta.json'};
for i = 1:length(files_to_delete)
    if exist(fullfile(savefolder,files_to_delete{i}),'file')
        delete(fullfile(savefolder,files_to_delete{i})); 
    end
end

%if ~exist(savefolder,'dir'), mkdir(savefolder); end

%read relevant meta-data from header
imageIdx = 1;
imgInfo = imfinfo(filename);
imgArtist = jsondecode(imgInfo(imageIdx).Artist);
imgDescription = splitlines(imgInfo(imageIdx).ImageDescription);
imgDescription = imgDescription(1:end-1);
imgSoftware = splitlines(imgInfo(imageIdx).Software);
for i = 1:length(imgSoftware)
    evalc(imgSoftware{i});
end
objResolution = SI.objectiveResolution;

if strcmp(imgInfo(imageIdx).ResolutionUnit,'Centimeter')
    %scaleF = 1/10;
    resLabel = Tiff.ResolutionUnit.Centimeter;
elseif     strcmp(imgInfo(imageIdx).ResolutionUnit,'Inch')
    %scaleF = 1;
    resLabel = Tiff.ResolutionUnit.Inch;
end
%XYres = [imgInfo(imageIdx).XResolution*scaleF,imgInfo(imageIdx).YResolution*scaleF];

%read image data
%rawData = loadFramesBuff(fn);
rawData = readTiffFast(filename,options); %this is a faster tiff loading function

%get circle centre offset
try
    windowCenter = SI.hDisplay.circleOffset;
catch
    windowCenter = [0 0];
end

%find centers, sizes and nLines of each FOV
si_rois_all = imgArtist.RoiGroups.imagingRoiGroup.rois;
si_rois = si_rois_all(logical([si_rois_all.enable]'));
nFOVs = numel(si_rois);
cXY = nan(2, nFOVs);
sXY = nan(2, nFOVs);
nLines = nan(nFOVs, 1);
for iFOV = 1:nFOVs
    cXY(:, iFOV) = si_rois(iFOV).scanfields.centerXY - 0.001*windowCenter';
    sXY(:, iFOV) = si_rois(iFOV).scanfields.sizeXY;
    nLines(iFOV) = si_rois(iFOV).scanfields(1).pixelResolutionXY(2);
end
nValidLines = sum(nLines);

%get stack info
Zs = SI.hStackManager.zs;
Zvals = unique(Zs); %NB this automatically sorts ascendingly
nZs = length(Zvals); %total number of depths defined
%Zdepths = diff(Zvals); %TO
%Zdepths = [Zdepths(1) Zdepths];

% reconstruct image in each plane
imgStack = [];
fprintf('Writing tiff slice nr. ');
for iZ = 1:nZs
    
    %display a iZ/nZ counter (and replace previous entry)
    if iZ>1
        for k=0:log10(iZ-1), fprintf('\b'); end
        for kk=0:log10(nZs-1), fprintf('\b'); end
        fprintf('\b')
    end
    fprintf('%d/%d', iZ, nZs);
    
    nLinesPerGap = (imgInfo(1).Height - nValidLines) / (nFOVs - 1);
    fovStartIdx = [1; cumsum(nLines(1:end-1) + nLinesPerGap) + 1];
    fovEndIdx = fovStartIdx + nLines - 1;
    for iFOV = 1:nFOVs
        rawImgs = permute(rawData(fovStartIdx(iFOV):fovEndIdx(iFOV), :, iZ:nZs:end),[2,1,3]);
        meanImg{iFOV} = mean(rawImgs,3,'native');
    end
    
    %stitch the FOVs together into a square grid
    [imgM,XX,YY] = stitchMultiFOV_adj(meanImg,cXY,sXY);
    imgStack(:,:,iZ) = imgM;
    
    imgM = imgM'; %need to transpose back for Tiff stack (not sure why...)

    %imgM = bitshift(imgM, 8); 
    
    %Write tiff
    if iZ == 1
        t = Tiff(fullfile(savefolder,savename),'w');
    else
        t = Tiff(fullfile(savefolder,savename),'a');
    end
    setTag(t,'Photometric',Tiff.Photometric.MinIsBlack);
    setTag(t,'Compression',Tiff.Compression.None);
    setTag(t,'BitsPerSample',16);
    setTag(t,'SamplesPerPixel',1);
    setTag(t,'SampleFormat',Tiff.SampleFormat.Int);
    setTag(t,'ResolutionUnit',resLabel);
    setTag(t,'XResolution',imgInfo(imageIdx).XResolution);
    setTag(t,'YResolution',imgInfo(imageIdx).YResolution);
    setTag(t,'ImageLength',size(imgM,1));
    setTag(t,'ImageWidth',size(imgM,2));
    setTag(t,'PlanarConfiguration',Tiff.PlanarConfiguration.Chunky);
    %setTag(t,'ImageDepth',Zdepths(iZ)/10e4); %this doesn't work...
    
    write(t,imgM);
    close(t);
    
    %OLD METHOD
%     if iZ  == 1
%         imwrite(imgM,fullfile(savefolder,savename),'Resolution',XYres,'Compression','none')
%     else
%         imwrite(imgM,fullfile(savefolder,savename),'Resolution',XYres,'Compression','none','WriteMode','append')
%     end
    
    %imagesc(XX, YY, imgM); hold on;
    
end

%re-name raw data
if ~strcmp(filename,fullfile(ff,['referenceImage.raw',fext]))
    movefile(filename,fullfile(ff,['referenceImage.raw',fext])); 
end

fprintf(' done!\n')

end


function [imgM,xVec,yVec] = stitchMultiFOV_adj(meanImg,cXY,sXY)
%WARNING: this only works for adjacent FOVs with equal spacing between
%lines

[nCols,nLines] = cellfun(@size,meanImg);

nx = sum(nCols);
ny = max(nLines);

tLeftXY = cXY - sXY/2;
bRightXY = cXY + sXY/2;

[xymin,xymax] = bounds([tLeftXY,bRightXY],2);
xVec = linspace(xymin(1),xymax(1),nx);
yVec = linspace(xymin(2),xymax(2),ny);

imgM = zeros(length(xVec),length(yVec),'int16');
for ii = 1:length(meanImg)
    [~,xIdx] = min(abs(xVec-tLeftXY(1,ii)));
    [~,yIdx] = min(abs(yVec-tLeftXY(2,ii)));
    imgM(xIdx+(1:nCols(ii))-1,yIdx+(1:nLines(ii))-1) = meanImg{ii};
end

end

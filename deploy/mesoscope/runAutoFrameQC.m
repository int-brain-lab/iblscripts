function [frameQC_frames,frameQC_names,badframes] = runAutoFrameQC(datPath, options)

if nargin < 2
    %initialize tiff reading options;
    options = {};
    options.frameStride = 12;
    options.firstFrame = 1;
    options.lastFrame = 0; %just for initialization, this will be updated in for-loop
end

if nargin <1
    datPath = 'M:\Subjects\SP035\2023-03-01\001\raw_imaging_data_02';
end

plot_flag = true;

fileList = dir(fullfile(datPath, '*.tif'));
nFiles = numel(fileList);

%% get validLines info from first tiff header (just to split lines)
fprintf('Reading first tiff header...\n');
fpath = fullfile(fileList(1).folder, fileList(1).name);
if options.lastFrame>0 %for testing on fewer frames than total
    lastFrameToRead= options.lastFrame;
    nFiles = min([nFiles,floor(lastFrameToRead/nFrames(fpath))]);
else
    lastFrameToRead = Inf;
end

fInfo = imfinfo(fpath);
fArtist = jsondecode(fInfo(1).Artist);
fSoftware = fInfo(1).Software;
evalc(fSoftware); %this builds the SI structure

si_rois_all = fArtist.RoiGroups.imagingRoiGroup.rois;
si_rois = si_rois_all(logical([si_rois_all.enable])); %only consider the rois that were 'enabled'
nrois = numel(si_rois);

Ly_all = arrayfun(@(x) x.scanfields(1).pixelResolutionXY(2),si_rois);

Zs = [si_rois.zs];
Zvals = unique(Zs);
nZs = length(Zvals);
for iZ = 1:length(Zs)
    Zidxs(iZ)=find(Zvals==Zs(iZ));
end

if length(SI.hChannels.channelSave)==1
    
    fprintf('Single channel detected...\n');
    
    %if ~SI.hStackManager.enable & nZs==1 %for single plane, single channel
        fprintf('No z-stack found, treating as single plane...\n');
        Ly = Ly_all;
        n_rows_sum = sum(Ly);
        n_flyback = (fInfo(1).Height - n_rows_sum) / max(1, (nrois - 1));
        irows = [0 cumsum(Ly'+n_flyback)]+1; %MATLAB indexing!
        irows(end) = [];
        irows(2,:) = irows(1,:) + Ly' - 1;
        validLines = {[]};
        for i = 1:nrois
            validLines{1} = [validLines{1} irows(1,i):(irows(2,i))];
        end
        fprintf('Found %i FOVs, %i valid lines out of %i.\n',nrois,length(validLines{1}),fInfo(1).Height);
        
    %elseif SI.hStackManager.enable & all([si_rois.discretePlaneMode]) %several discrete planes, single channel
        
        fprintf('Treating as z-stack with multiple discrete planes...\n');
        
        %FOV = struct('lineIdx',double.empty(0,nrois));
        nFOVs = zeros(1,nZs); %nr of FOVs per z-plane
        n_rows_sum = zeros(1, nZs);
        irows = double.empty(2,0);
        islices = double.empty(1,0);
        validLines = {};
        for iZ = 1:nZs
            % get FOV info for each slice in the z-stack
            nFOVs(iZ) = sum(Zidxs==iZ); %nr of FOVs in this slice
            Ly = Ly_all(Zidxs==iZ); %indexes of valid lines in this slice
            n_rows_sum(iZ) = sum(Ly); %nr of valid lines in this slice
            n_flyback = (fInfo(1).Height - n_rows_sum(iZ)) / max(1, (nFOVs(iZ) - 1)); %nr of flyback/flyto lines
            irow = [0 cumsum(Ly'+n_flyback)]+1; %MATLAB indexing!
            irow(end) = [];
            irow(2,:) = irow(1,:) + Ly' - 1;
            irows = horzcat(irows,irow);
            
            iFOVs = find(Zidxs==iZ); %the FOVs that are contained in this plane
            islices = horzcat(islices,iFOVs(1:nFOVs(iZ)));
            
            validLines{iZ} = [];
            for i = 1:nFOVs(iZ)
                validLines{iZ} = [validLines{iZ} irow(1,i):(irow(2,i))];
            end
            fprintf('Slice %i: Found %i FOVs, %i valid lines out of %i.\n',islices(iZ),nFOVs(iZ),length(validLines{iZ}),fInfo(1).Height);
            
        end
    %end
    
    
else
    
    error('Could not figure out how the tiff is structured, write arrays manually.');
    
end




%% load tiff-stacks (one for each slice) and get stats

%update frame stride after with slice factor
frameStride = options.frameStride; 
options.frameStride = frameStride*nZs; %overwrite to reflect multi-plane

%make array of all slice indexes across files
nFramesInFirstFile = nFrames(fullfile(fileList(1).folder, fileList(1).name));
nVolFramesMax = ceil((nFramesInFirstFile*nFiles)/nZs);
sliceIdx = repmat([1:nZs],[1,nVolFramesMax]);

nVolFramesAfterStride = ceil(nVolFramesMax/frameStride); %maximum nr (will need to chop off end)

%initialize 
meanImg_eachFOV = cell(1,nrois);
meanTrace_eachFOV = nan(nrois,nVolFramesAfterStride);
maxTrace_eachFOV = nan(nrois,nVolFramesAfterStride);
medianTrace_eachFOV = nan(nrois,nVolFramesAfterStride);

meanTrace_all = []; 
stdTrace_all = [];
maxTrace_all = []; 
medianTrace_all = []; 

for iZ = 1:nZs

%islice = Zvals(iZ);
fprintf('\nSlice nr. %i\n',iZ);

%initialize variables
frames_sampled = [];
volFrames_sampled = [];
lastframe_all = 0;
volframe_all = 0;

TotNumFrames = 0;
superFrames_cnt = 0;
%median_stacks = nan(nFiles,1);
frameBounds_stacks = nan(nFiles,2);

if frameStride>1
    fprintf('Running QC across every %ith frame in tiff stack: \nLoading file nr. ',frameStride);
else
    fprintf('Running QC across every frame in tiff stack: \nLoading file nr. ',frameStride);
end

for iFile = 1:nFiles
    
    %iFile/nFiles counter
    if iFile>1
        for k=0:log10(iFile-1), fprintf('\b'); end
        for kk=0:log10(nFiles-1), fprintf('\b'); end
        fprintf('\b')
    end
    fprintf('%d/%d', iFile, nFiles);
    
    fpath = fullfile(fileList(iFile).folder, fileList(iFile).name);
    
    FramesInFile = nFrames(fpath);
    TotNumFrames = TotNumFrames + FramesInFile;
    options.lastFrame = min(lastFrameToRead,FramesInFile);
    lastFrameToRead = lastFrameToRead-FramesInFile;
    
    %make vector of frame indexes
    iSlice = sliceIdx(lastframe_all+iZ);
    framenrs = (iSlice+options.firstFrame-1):options.frameStride:options.lastFrame;
    totFrames_idx = lastframe_all+framenrs;
    frames_sampled = [frames_sampled totFrames_idx];
    
    volframenrs = 1:frameStride:length(framenrs)*frameStride;
    volFrames_idx = volframe_all+volframenrs;
    volFrames_sampled = [volFrames_sampled volFrames_idx];
    
    frameBounds_stacks(iFile,:) = [lastframe_all+options.firstFrame, lastframe_all+options.lastFrame];
    
    lastframe_all = lastframe_all+FramesInFile;
    volframe_all = volframe_all+ceil(FramesInFile/nZs);
    
    superFrames_idx = superFrames_cnt + [1:length(framenrs)];
    superFrames_cnt = superFrames_cnt + length(framenrs);
    
    %get flattened stack 
    %stack_full = ScanImageTiffReader(datpath).data();
    stack_full = readTiffFast(fpath,options); %this is a faster tiff loading function
    
    
    % for convenience, only consider valid lines of first slice for overall stats
    if iZ==1
        
        stack = stack_full(validLines{iZ},:,:);
        stack_flat = single(reshape(stack,[size(stack,1)*size(stack,2),size(stack,3)]));
        
        %get some stats across all pixels of stack
        trace_median = median(stack_flat,1);
        trace_mean = mean(stack_flat,1);
        trace_std = std(stack_flat,[],1);
        trace_max = max(stack_flat,[],1);
        %median_stacks(iFile) = median(trace_mean);
        
        %concatenate with stats from previous tiffs
        meanTrace_all = [meanTrace_all trace_mean];
        medianTrace_all = [medianTrace_all, trace_median];
        stdTrace_all = [stdTrace_all trace_std];
        maxTrace_all = [maxTrace_all trace_max];
        
    end
    
    %get info for each FOV
    nrois = nFOVs(iZ); %length(islices==iSlice);
    iFOVs = find(Zidxs==iZ); %the FOVs that are contained in this plane
    traceFOV_median = nan(nrois,length(framenrs));
    traceFOV_mean = nan(nrois,length(framenrs));
    traceFOV_max = nan(nrois,length(framenrs));
    for i = 1:nrois
        stackFOV = stack_full(irows(1,iFOVs(i)):irows(2,iFOVs(i)),:,:);
        meanImgFOV = squeeze(mean(stackFOV,3));
        stackFOV_flat = single(reshape(stackFOV,[size(stackFOV,1)*size(stackFOV,2),size(stackFOV,3)]));
        try
            traceFOV_median(i,:) = nanmedian(stackFOV_flat,1);
            traceFOV_mean(i,:) = nanmean(stackFOV_flat,1);
            traceFOV_max(i,:) = nanmax(stackFOV_flat,[],1);
        catch
            if any(isnan(stackFOV_flat(:)))
                error('NaNs found in image stack... Cannot do stats')
            end
            traceFOV_median(i,:) = median(stackFOV_flat,1);
            traceFOV_mean(i,:) = mean(stackFOV_flat,1);
            traceFOV_max(i,:) = max(stackFOV_flat,[],1);
        end
        meanImg_eachFOV{iFOVs(i)} = cat(3,meanImg_eachFOV{iFOVs(i)},meanImgFOV);
    end
    medianTrace_eachFOV(iFOVs,superFrames_idx) = traceFOV_median;
    meanTrace_eachFOV(iFOVs,superFrames_idx) = traceFOV_mean;
    maxTrace_eachFOV(iFOVs,superFrames_idx) = traceFOV_max;
    
    
end

fprintf('\n');

end

medianTrace_eachFOV(:,superFrames_idx(end)+1:end) = [];
meanTrace_eachFOV(:,superFrames_idx(end)+1:end) = [];
maxTrace_eachFOV(:,superFrames_idx(end)+1:end) = [];

%% plot some stuff

if plot_flag
    
    %plot max, mean and median traces
    figure('Name',[fileList(1).name(1:end-16) ', average traces']);
    set(gcf,'Units','normalized','Position',[0.05 0.65 0.05+0.05*nFiles 0.3]);
    
    ax(1) = subplot(3,1,1);
    hold on;
    plot(volFrames_sampled,maxTrace_eachFOV,'linewidth',1);
    plot(volFrames_sampled,maxTrace_all,'k','linewidth',2)
    xlabel('Frame nr.');
    ylabel('max F');
    
    ax(2) = subplot(3,1,2);
    hold on;
    plot(volFrames_sampled,meanTrace_eachFOV,'linewidth',1);
    plot(volFrames_sampled,meanTrace_all,'k','linewidth',2)
    xlabel('Frame nr.');
    ylabel('mean F');
    
    ax(3) = subplot(3,1,3);
    hold on;
    plot(volFrames_sampled,medianTrace_eachFOV,'linewidth',1);
    plot(volFrames_sampled,medianTrace_all,'k','linewidth',2)
    xlabel('Frame nr.');
    ylabel('median F');
    %legend()
    
    linkaxes(ax,'x');
    xlim([0,TotNumFrames])
    
    c = colororder;
    
    %plot each FOV mean image across stacks (either downsampled full img or central patch)
    dsFactor = 4;
    boxSize = 100;
    k=0;
    stacksplotted = round(linspace(1,nFiles,6)); %plot first, last and 4 in between
    figure('Name',[fileList(1).name(1:end-16) ', mean images']);
    set(gcf,'Units','normalized','Position',[0.05 0.05 0.05+0.05*length(stacksplotted) 0.05+0.08*nrois]);
    for i=1:nrois
        sz = size(meanImg_eachFOV{i});
        for j=stacksplotted
            k=k+1;
            ax2(k)=subplot(nrois,length(stacksplotted),k);
            %imagesc(squeeze(meanImg_eachFOV{i}(1:dsFactor:end,1:ds_factor:end,j)));
            imagesc(squeeze(meanImg_eachFOV{i}((1:boxSize)+floor((sz(1)-boxSize)/2),(1:boxSize)+floor((sz(2)-boxSize)/2),j)));
            colormap('gray');
            axis square
            caxis([0,5000]);
            set(gca,'xtick',[],'ytick',[]);
            if j==1
                ylabel(sprintf('fov%0.2d',i-1),'FontWeight','bold','Color',c(i,:));
            end
            if i==1
                title([num2str(frameBounds_stacks(j,1)),'-',num2str(frameBounds_stacks(j,2))],'Color','w');
            end
        end
    end
    linkaxes(ax2);
    try
        set_bb;
    end
    
end

%% run metrics

%define QC types
frameQC_names = {'ok','PMT off','galvos fault','high signal'};
frameQC_frames = zeros(1,TotNumFrames);
badframes = []; %by default there are no badframes and all frames are 'ok'

%first we find outlier frames
tr = medianTrace_all;
fr = volFrames_sampled;
fr_all = 1:TotNumFrames;
st = options.frameStride;

C = median(tr);
mad = median(abs(tr-C));
outlier_thresh_L = 5; %in MADs
outlier_thresh_U = 10; %in MADs

L = C-outlier_thresh_L*mad;
U = C+outlier_thresh_U*mad;
outliers = tr<L | tr>U;

outliers_low = tr<L; %these are probably PMT off or galvos fault
outliers_high = tr>U; %these are probably light artefacts (NOT TESTED)

%now we divide outlier frames into epochs and categorize each into one of
%several possible QC events

%low outlier frames with abnormally low variance in maxTrace are 'PMT off',
%otherwise we assume they are 'galvos fault' for now.
%(normal variance defined here as at >0.2*MADs of non-outlier maxTrace)
mad_max_ref = median(abs(maxTrace_all(~outliers)-median(maxTrace_all(~outliers))));
vals = outliers_low;
vals(end) = false; %just so outliers at end of recording have an epoch 'end'
outlierEpochs = [find(diff([0,vals])>0); find(diff([0,vals])<0)];
outlierEpochs_fr = [fr(diff([0,vals])>0); fr(diff([0,vals])<0)];
for i = 1:size(outlierEpochs,2)
    trO = maxTrace_all([outlierEpochs(1,i):outlierEpochs(2,i)]);
    mad_max_outlier = median(abs(trO-median(trO)));
    QCblock_start = max([1,outlierEpochs_fr(1,i)-st]); %one strideLength before auto-detected outlier start (with exception for first frame)
    QCblock_end = min([fr(end),outlierEpochs_fr(2,i)+st]); %one strideLength after auto-detected outlier start (with exception for final frame)
    if mad_max_outlier<0.2*mad_max_ref
        frameQC_frames(QCblock_start:QCblock_end) = 1;
    else
        frameQC_frames(QCblock_start:QCblock_end) = 2;
    end
end

%high outlier frames with abnormally high median are 'high signal'
vals = outliers_high;
vals(end) = false; %just so outliers at end of recording have an epoch 'end'
outlierEpochs = [find(diff([0,vals])>0); find(diff([0,vals])<0)];
outlierEpochs_fr = [fr(diff([0,vals])>0); fr(diff([0,vals])<0)];
for i = 1:size(outlierEpochs,2)
    iO = vals([outlierEpochs(1,i):outlierEpochs(2,i)]);
    QCblock_start = max([1,outlierEpochs_fr(1,i)-st]); %one strideLength before auto-detected outlier start (with exception for first frame)
    QCblock_end = min([fr(end),outlierEpochs_fr(2,i)+st]); %one strideLength after auto-detected outlier start (with exception for final frame)
    if any(iO)
        frameQC_frames(QCblock_start:QCblock_end) = 3;
    end
end

%SOME MORE IDEAS:
%find slow drift in fluorescence: meanTrace drifting
%find potential z-drift: meanImg cross-correlations between first and last stack?
%find frames with sudden high signal ('light artefact')

%log all frameQC_frames as badframes for suite2p
badframes = find(frameQC_frames>0)-1; %because of 0-indexing

if sum(outliers>0)
    
    if plot_flag & sum(outliers>0)

        figure('Name',[fileList(1).name(1:end-16) ', Frame QC'],'Units','normalized','Position',[0.45 0.6 0.5 0.3]);
        h(1) = plot(fr,tr,'Color',[.5 .5 .5],'DisplayName','MedianTrace');
        hold on
        h(2) = plot(fr(outliers),tr(outliers),"x",'DisplayName','outliers');
        frQC_vals = unique(frameQC_frames);
        for i=1:length(unique(frameQC_frames))
            h(2+i) = plot(fr_all(frameQC_frames==frQC_vals(i)),zeros(1,sum(frameQC_frames==frQC_vals(i))),'o','DisplayName',frameQC_names{frQC_vals(i)+1});
        end
        yline(U,':','U Threshold');
        yline(L',':','L Threshold');
        yline(C,':','Central Value');
        xlabel('Frame nr')
        ylabel('F');
        xlim([0,TotNumFrames])
        legend(h);
    end
    
else
    
    fprintf('No badframes detected!\n');
    
end

end
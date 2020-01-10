function run_batch_ks2_ibl

%% SET PATHS - also for vizualisation
addpath(genpath('~/Documents/MATLAB/Kilosort2')) % path to kilosort folder
addpath('~/Documents/MATLAB/npy-matlab/npy-matlab');
addpath(genpath('/home/ibladmin/Documents/MATLAB/spikes'));
addpath('/home/ibladmin/Documents/PYTHON/iblscripts/deploy/serverpc/kilosort2');

rootH = '/mnt/h0/';
figure_dir = '~/Documents/figures';
set(groot, 'defaultaxesfontsize', 6);

%% ========== set path to files
% rootZ is the directory containing the raw AP traces, one probe per folder
% rootH is the scratch directory, SSD drive, to memmap binary data
ROOTZ = {...
    '/mnt/s0/Data/Subjects/CSHL049/2020-01-09/001/raw_ephys_data/probe00',... # running
   '/mnt/s0/Data/Subjects/CSHL049/2020-01-09/001/raw_ephys_data/probe01',... # running
    };

%% ========== run spike sorting
for m = 1:length(ROOTZ)
    
    close all;
    rootZ = ROOTZ{m};
    a = strsplit(rootZ, '/');
    
    % ==================== % wait for the transfer to finish
    while ~exist(sprintf('%s/spike_sorting.flag', rootZ), 'file')
        pause(60);
    end
    disp(rootZ);
    
    %% ========== %
    if ~exist(sprintf('%s/whitening_mat.npy', rootZ), 'file')
        
        % ==================== % run KS2
        run_ks2_ibl(rootZ, rootH);
        
        % ==================== % save the ks2 figures
        figure(1); tightfig;
        print(gcf, '-dpdf', sprintf('%s/%s_%s_%s_%s_ksFig1.pdf', figure_dir, a{6}, a{7}, a{8}, a{10}));
        figure(2); tightfig;
        print(gcf, '-dpdf', sprintf('%s/%s_%s_%s_%s_ksFig2.pdf', figure_dir, a{6}, a{7}, a{8}, a{10}));
    end
    
    %% ===================== plots: from https://github.com/cortex-lab/spikes/
    close all;
    
    % plot drift map
    [spikeTimes, spikeAmps, spikeDepths, spikeSites] = ksDriftmap(rootZ);
    clf; plotDriftmap(spikeTimes, spikeAmps, spikeDepths);
    title(sprintf('%s/%s/%s/%s', a{6}, a{7}, a{8}, a{10}), 'interpreter', 'none');
    axis tight; axisNotSoTight; tightfig;
    print(gcf, '-dpdf', sprintf('%s/%s_%s_%s_%s_ksDriftmap.pdf', figure_dir, a{6}, a{7}, a{8}, a{10}));
    
    % spike amplitudes
    [pdfs, cdfs] = computeWFampsOverDepth(spikeAmps, spikeDepths, 0:30:min(max(spikeAmps),800), ...
        0:40:3840, spikeTimes(end));
    plotWFampCDFs(pdfs, cdfs, 0:30:min(max(spikeAmps),800), 0:40:3840);
    print(gcf, '-dpdf', sprintf('%s/%s_%s_%s_%s_splikeAmplitudes.pdf', figure_dir, a{6}, a{7}, a{8}, a{10}));
    
    % LFP
    lfpD = dir(fullfile(rootZ, '*.lf.bin')); % LFP file from spikeGLX specifically
    if ~isempty(lfpD),
        lfpFilename = fullfile(rootZ, lfpD(1).name);
        lfpFs = 2500;  % neuropixels phase3a
        nChansInFile = 385;  % neuropixels phase3a, from spikeGLX
        [lfpByChannel, allPowerEst, F, allPowerVar] = ...
            lfpBandPower(lfpFilename, lfpFs, nChansInFile, []);
        chanMap = readNPY(fullfile(rootZ, 'channel_map.npy'));
        nC = length(chanMap);
        allPowerEst = allPowerEst(:,chanMap+1)'; % now nChans x nFreq
        % plot LFP power
        dispRange = [0 100]; % Hz
        marginalChans = [10:50:nC];
        freqBands = {[1.5 4], [4 10], [10 30], [30 80], [80 200]};
        plotLFPpower(F, allPowerEst, dispRange, marginalChans, freqBands);
        tightfig;
        print(gcf, '-dpdf', sprintf('%s/%s_%s_%s_%s_LFP.pdf', figure_dir, a{6}, a{7}, a{8}, a{10}));
    end
    
end

%% ========== start extraction

for m = 1:length(ROOTZ)
    rootZ = ROOTZ{m};
    a = strsplit(rootZ, '/');
    pathnames{m} = fullfile(a{1:end-2});
end

% now take only the unique ones - do once for each session only
pathnames = unique(pathnames);

for p = 1:length(pathnames),
    % run the python command that will do the extraction
    command = ['cd ~/Documents/PYTHON/iblscripts/deploy/serverpc/utils; ' ...
        'source ~/Documents/PYTHON/envs/iblenv/bin/activate; pwd; ' ...
        'python extract_ephys_manual.py /' pathnames{p}];
    [status,cmdout] = system(command)
end

end


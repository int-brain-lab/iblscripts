function run_batch_ks2_ibl(rootH, dry)
% run_batch_ks2_ibl(rootH, dry)
% rootH: swap directory (defaults to '/mnt/h0/') SSD drive that will be used to host temporary binary files for KS2
% dry: bool (defaults to false). If true, just lists the probe files that would be spike sorted by the script
% Author: Anne Urai

%% SET PATHS - also for vizualisation
addpath(genpath('~/Documents/MATLAB/Kilosort2')) % path to kilosort folder
addpath('~/Documents/MATLAB/npy-matlab/npy-matlab');
addpath(genpath('/home/ibladmin/Documents/MATLAB/spikes'));
addpath('/home/ibladmin/Documents/PYTHON/iblscripts/deploy/serverpc/kilosort2');

if nargin <= 0, rootH = '/mnt/h0/'; end
if nargin <= 0, dry = false; end

root = '/mnt/s0/Data/Subjects';
figure_dir = '~/Documents/figures';
set(groot, 'defaultaxesfontsize', 6);

%% ========== set path to files
% rootZ is the directory containing the raw AP traces, one probe per folder
% rootH is the scratch directory, SSD drive, to memmap binary data
allfolders = dir(sprintf('%s/*/20*/0*/raw_ephys_data/probe*/*.ap.cbin', root));
[~, idx] = sort([allfolders(:).datenum], 'descend'); % start with most recent
allfolders = allfolders(idx);

to_spikesort = {};
for f = 1:length(allfolders),
    spikesorting_present = dir(sprintf('%s/spike_sorting_ks2.log', allfolders(f).folder));
    if isempty(spikesorting_present),
        to_spikesort{end+1} = allfolders(f).folder;
    end
end

disp(to_spikesort');
if dry, return, end

%% ========== run spike sorting
for m = 1:length(to_spikesort)

    close all;
    rootZ = to_spikesort{m};
    a = strsplit(rootZ, '/');
    fig_name = sprintf('%s_%s_%s_%s', a{6}, a{7}, a{8}, a{10});
    disp(rootZ);




    %% ==================== % run KS2
    run_ks2_ibl(scratch_dir, rootH);

    %% ========== move files to correct location
    results_files = dir('/mnt/h0/temp/');
    for ff = 1:length(results_files),
        if ~contains(results_files(ff).name, 'ap.bin') && ~results_files(ff).isdir,
            copyfile(fullfile(results_files(ff).folder, results_files(ff).name), ...
                fullfile(rootZ, results_files(ff).name));
        end
    end

    %% ==================== % save the ks2 figures
    try
        figure(1);
        print(gcf, '-dpdf', sprintf('%s/%s_ksFig1.pdf', figure_dir, fig_name));
        figure(2);
        print(gcf, '-dpdf', sprintf('%s/%s_ksFig2.pdf', figure_dir, fig_name));

        % ===================== plots: from https://github.com/cortex-lab/spikes/
        % plot drift map
        [spikeTimes, spikeAmps, spikeDepths, spikeSites] = ksDriftmap(rootZ);
        clf; plotDriftmap(spikeTimes, spikeAmps, spikeDepths);
        title(sprintf('%s/%s/%s/%s', a{6}, a{7}, a{8}, a{10}), 'interpreter', 'none');
        axis tight;
        print(gcf, '-dpdf', sprintf('%s/%s_ksDriftmap.pdf', figure_dir, fig_name));

        % spike amplitudes
        [pdfs, cdfs] = computeWFampsOverDepth(spikeAmps, spikeDepths, 0:30:min(max(spikeAmps),800), ...
            0:40:3840, spikeTimes(end));
        plotWFampCDFs(pdfs, cdfs, 0:30:min(max(spikeAmps),800), 0:40:3840);
        print(gcf, '-dpdf', sprintf('%s/%s_splikeAmplitudes.pdf', figure_dir, fig_name));

        % LFP
        lfpD = dir(fullfile(rootZ, '*.lf.cbin')); % LFP file from spikeGLX specifically
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
            print(gcf, '-dpdf', sprintf('%s/%s_LFP.pdf', figure_dir, fig_name));
        end
    catch
        warning('Could not plot figures. Do you have the spikes repo from the cortexlab github on your path?');
    end
end

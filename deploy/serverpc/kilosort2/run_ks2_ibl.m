function run_ks2_ibl(rootZ, rootH, dir_pattern, channel_map_file)
% rootZ is the directory containing the raw AP traces, one probe per folder
% rootH is the scratch directory, SSD drive, to memmap binary data
%
% Example below:
% rootZ = '/mnt/s0/Data/Subjects/ZM_1150/2019-05-07/001/raw_ephys_data/probe_right';
% rootH = '/mnt/h0';  # temporary folder if different than main
% run_ks2_ibl(rootZ, rootH)
% run_ks2_ibl(rootZ, rootH, dir_pattern, channel_map_file)

try
    %% 1) Set paths and get ks2 commit hash
    addpath(genpath('~/Documents/MATLAB/Kilosort2')) % path to kilosort folder
    addpath('~/Documents/MATLAB/npy-matlab/npy-matlab')
    [~, hash] = unix('git --git-dir ~/Documents/MATLAB/Kilosort2/.git rev-parse --verify HEAD');
    disp(["ks2 version: " hash])
    
    %% 2) Parse input arguments
    if nargin <= 1, rootH = rootZ; end
    if nargin <= 2, dir_pattern = '*.ap.bin'; end
    if nargin <= 3, channel_map_file = '~/Documents/MATLAB/Kilosort2/configFiles/neuropixPhase3A_kilosortChanMap.mat'; end

    %% 3) get IBL params
    ops = ibl_ks2_params;
    
    %% 4) KS2 run
    fprintf('Looking for data inside %s \n', rootZ)
    
    % is there a channel map file in this folder?
    fs = dir(fullfile(rootZ, 'chan*.mat'));
    if ~isempty(fs)
        ops.chanMap = fullfile(rootZ, fs(1).name);
    end
    
    % find the binary file
    ops.fbinary = fullfile(rootZ, getfield(dir(fullfile(rootZ, dir_pattern)), 'name'));
    
    % preprocess data to create temp_wh.dat
    rez = preprocessDataSub(ops);
    
    % time-reordering as a function of drift
    rez = clusterSingleBatches(rez);
    save(fullfile(rootZ, 'rez.mat'), 'rez', '-v7.3');
    
    % main tracking and template matching algorithm
    rez = learnAndSolve8b(rez);
    
    % final merges
    rez = find_merges(rez, 1);
    
    % final splits by SVD
    rez = splitAllClusters(rez, 1);
    
    % final splits by amplitudes
    rez = splitAllClusters(rez, 0);
    
    % decide on cutoff
    rez = set_cutoff(rez);
    
    fprintf('found %d good units \n', sum(rez.good>0))
    
    % write to Phy
    fprintf('Saving results to Phy  \n')
    rezToPhy(rez, rootZ);
    
    %% 5) WRAP-UP
    fid = fopen([rootZ filesep 'spike_sorting_ks2.log'], 'w+');
    for ff = fieldnames(ops)'
        val = ops.(ff{1});
        if isnumeric(val) | islogical(val)
            str = mat2str(val);
        else
            str = val;
        end
        fwrite(fid,['ops.' ff{1} ' = ' str ';' newline]);
    end
    fclose(fid);
    
catch
    a = lasterror;
    str=[a.message newline];
    for m=1:length(a.stack)
        str = [str 'Error in ' a.stack(m).file ' line : ' num2str(a.stack(m).line) newline];
    end
    disp(str)
    disp('run_ks2_ibl.m failed') % this is used to find out that matlab failed in stdout
end

    function ops = ibl_ks2_params
        ops.commitHash = strip(hash);
        ops.chanMap = channel_map_file;
        ops.fs = 30000;   % sample rate
        ops.fshigh = 300;    % frequency for high pass filtering (150)
        ops.minfr_goodchannels = 0;  % minimum firing rate on a "good" channel (0 to skip)
        ops.Th = [10 4];   % threshold on projections (like in Kilosort1, can be different for last pass like [10 4])
        ops.lam = 10;  % how important is the amplitude penalty (like in Kilosort1, 0 means not used, 10 is average, 50 is a lot)
        ops.AUCsplit = 0.9; % splitting a cluster at the end requires at least this much isolation for each sub-cluster (max = 1)
        ops.minFR = 1/50; % minimum spike rate (Hz), if a cluster falls below this for too long it gets removed
        ops.momentum = [20 400]; % number of samples to average over (annealed from first to second value)
        ops.sigmaMask = 30; % spatial constant in um for computing residual variance of spike
        ops.ThPre = 8;  % threshold crossings for pre-clustering (in PCA projection space)
        ops.CAR = true;  % Common Average Referencing (median)
        % DANGER ZONE: changing these settings can lead to fatal errors
        % options for determining PCs
        ops.spkTh           = -6;      % spike threshold in standard deviations (-6)
        ops.reorder         = 1;       % whether to reorder batches for drift correction.
        ops.nskip           = 25;  % how many batches to skip for determining spike PCs
        ops.GPU                 = 1; % has to be 1, no CPU version yet, sorry
        % ops.Nfilt               = 1024; % max number of clusters
        ops.nfilt_factor        = 4; % max number of clusters per good channel (even temporary ones)
        ops.ntbuff              = 64;    % samples of symmetrical buffer for whitening and spike detection
        ops.NT                  = 32*1024+ ops.ntbuff; % 64*1024+ ops.ntbuff; % must be multiple of 32 + ntbuff. This is the batch size (try decreasing if out of memory).
        ops.whiteningRange      = 32; % number of channels to use for whitening each channel
        ops.nSkipCov            = 25; % compute whitening matrix from every N-th batch
        ops.scaleproc           = 200;   % int16 scaling of whitened data
        ops.nPCs                = 3; % how many PCs to project the spikes into
        ops.useRAM              = 0; % not yet available
        ops.trange = [0 Inf]; % time range to sort
        ops.NchanTOT    = 384; % total number of channels in your recording
        % you need to change most of the paths in this block
        ops.fproc       = fullfile(rootH, 'temp_wh.dat'); % proc file on a fast SSD
        ops.trange = [0 Inf]; % time range to sort
        ops.NchanTOT    = 385; % total number of channels in your recording
    end
end

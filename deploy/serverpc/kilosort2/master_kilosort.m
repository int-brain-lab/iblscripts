%% you need to change most of the paths in this block

addpath(genpath('/home/ibladmin/Documents/MATLAB/Kilosort2')) % path to kilosort folder
addpath('/home/ibladmin/Documents/MATLAB/npy-matlab/npy-matlab')

pathToYourConfigFile = '/home/ibladmin/Documents/MATLAB/ks2configFiles'; % take from Github folder and put it somewhere else (together with the master_file)
run(fullfile(pathToYourConfigFile, 'configFile384.m'))

rootH = '/mnt/h0';
ops.fproc       = fullfile(rootH, 'temp_wh.dat'); % proc file on a fast SSD
ops.chanMap = fullfile(pathToYourConfigFile, 'neuropixPhase3A_kilosortChanMap.mat');

ops.trange = [0 Inf]; % time range to sort
ops.NchanTOT    = 385; % total number of channels in your recording

% the binary file is in this folder
rootZ = '/mnt/s0/Data/Subjects/ZM_1150/2019-05-07/001/raw_ephys_data/probe_right';
rootZ = '/mnt/s0/Data/Subjects/ZM_1150/2019-05-02/001/raw_ephys_data/probe_right';

%% this block runs all the steps of the algorithm
fprintf('Looking for data inside %s \n', rootZ)

% is there a channel map file in this folder?
fs = dir(fullfile(rootZ, 'chan*.mat'));
if ~isempty(fs)
    ops.chanMap = fullfile(rootZ, fs(1).name);
end

% find the binary file
% fs          = [dir(fullfile(rootZ, '*.bin')) dir(fullfile(rootZ, '*.dat'))];
% ops.fbinary = fullfile(rootZ, fs(1).name);
ops.fbinary = fullfile(rootZ, getfield(dir(fullfile(rootZ, '*.imec.ap.bin')), 'name'))

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

%% if you want to save the results to a Matlab file... 

% discard features in final rez file (too slow to save)
rez.cProj = [];
rez.cProjPC = [];

% save final results as rez2
fprintf('Saving final results in rez2  \n')
fname = fullfile(rootZ, 'rez2.mat');
save(fname, 'rez', '-v7.3');


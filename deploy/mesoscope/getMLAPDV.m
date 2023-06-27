function [mlapdv,brainLocIds] = getMLAPDV(sess,varargin)

%this script pulls MLAPDV + brainLocationIds guesstimates of ROI centroids
%from pixel-maps, and writes those to the alf folder
%written by Samuel Picard (June 2023)

defaultRoot = 'M:\Subjects\';

p = inputParser;
validInput = @(x) isstring(x) | ischar(x);
addRequired(p,'sess',validInput);
addOptional(p,'date','',validInput);
addOptional(p,'session','',validInput);
addParameter(p,'root',defaultRoot,validInput);
parse(p,sess,varargin{:});

if ~isempty(strfind(p.Results.sess,filesep)) %assume we provided a session path
    sessionpath = sess;
    splitPath = split(sessionpath,'\');
    subject = splitPath{1};
    date = splitPath{2};
    session = splitPath{3};
else
    subject = p.Results.sess;
    date = p.Results.date;
    session = p.Results.session;
end
root = p.Results.root;
   
dataPath = fullfile(root,subject,date,session);
fns = dir(fullfile(dataPath,'alf','FOV*'));

for iFOV=1:length(fns)

    alf_data_path = fullfile(dataPath,'alf',fns(iFOV).name);
    
    %load neuron centroids in pixel space
    stackPos = readNPY(fullfile(alf_data_path,'mpciROIs.stackPos.npy'));
    
    %load mlapdv + brainLocIds maps of pixels
    map_mlapdv = readNPY(fullfile(alf_data_path,'mpciMeanImage.mlapdv_estimate.npy'));
    map_brainLocIds = readNPY(fullfile(alf_data_path,'mpciMeanImage.brainLocationIds_estimate.npy'));

    %get centroid MLAPDV + brainID by indexing pixel-map with centroid locations
    mlapdv = double(nan(size(stackPos)));
    brainLocIds = int16(nan(1,size(stackPos,1)));
    for iROI=1:length(stackPos)
        mlapdv(iROI,:) = map_mlapdv(stackPos(iROI,1), stackPos(iROI,2), :);
        brainLocIds(iROI) = int16(map_brainLocIds(stackPos(iROI,1), stackPos(iROI,2)));
    end
        
    %write mlapdv + brainLocIds of ROIs to disk
    writeNPY(mlapdv,fullfile(alf_data_path,'mpciROIs.mlapdv_estimate_test.npy'));
    writeNPY(brainLocIds,fullfile(alf_data_path,'mpciROIs.brainLocationIds_estimate_test.npy'));
    
end
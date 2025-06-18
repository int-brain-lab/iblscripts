% this looks for all _ibl_rawImagingData.meta.json files, and re-runs
% metadata extraction for those sessions (and their reference stacks)

rootdir = 'Y:\Subjects\';
fn = '_ibl_rawImagingData.meta.json';

%open an alyx session
alyx = Alyx();

%search for all files
filelist = dir(fullfile(rootdir, ['**\',fn]));  
filelist = filelist(~[filelist.isdir]);  %remove folders from list

%% run across all sessions
failedSessions = false(1,length(filelist));
failedRefs = false(1,length(filelist));
for i= 1:length(filelist)
    ExpRef = filelist(i).folder;
    try
        meta = mesoscopeMetadataExtraction(ExpRef,'alyx',alyx);
    catch %ME
        failedSessions(i) = true;
        warning(sprintf('Something was wrong, skipping this session.\n'));
    end
    try
        %subfolders that may contain reference images/stacks:
        fn_refs = {'reference','hirez'}; 
        %fn_refs = {'reference'}; %ignore hirez stacks for now (they take long)
        for iF = 1:length(fn_refs)
            if exist(fullfile(ExpRef,fn_refs{iF}),'dir')
                mesoscopeMetadataExtraction_ref(fullfile(ExpRef,fn_refs{iF}),'alyx',alyx);
            end
        end
    catch %ME
        failedRefs(i) = true;
        warning(sprintf('Something was wrong, skipping this session.\n'));
    end
end
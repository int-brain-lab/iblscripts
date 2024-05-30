function out = mesoscopePostExpt(ExpRefs)

%this function is to be run after a mesoscope experiment at the scanimage PC.
%1) TODO look for all of today's data
%2) write Frame QC
%3) extract metadata
%4) TODO copy data to server

%for testing
% if nargin<1
%     ExpRefs = {'Y:\Subjects\SP052\2024-01-10\002\raw_imaging_data_00'};
% end

%make sure we are logged into alyx
if exist('alyx','var')
    if isa(alyx,'Alyx')
        if ~alyx.IsLoggedIn
            alyx = alyx.login;
        end
    else
        alyx = Alyx();
    end
else
    alyx = Alyx();
end

%% 1) TODO look for all of today's data
localPath = 'F:\ScanImageAcquisitions';
%...

%% 2) write Frame QC
for iExpt = 1:length(ExpRefs)
    ExpRef = {};
    try
        if isempty(dir(fullfile(ExpRefs{iExpt},'*2P*.tif')))
            raw_folders = dir(fullfile(ExpRefs{iExpt},'raw_imaging_data*'));
            for i = 1:length(raw_folders)
                ExpRef{i}= fullfile(ExpRefs{iExpt},raw_folders(i).name);
            end
        else
            ExpRef = ExpRefs(iExpt);
        end
        for i = 1:length(ExpRef)
            writeFrameQC(ExpRef{i},'auto');
        end
    catch ME
        rethrow(ME)
    end
end

%% 3) extract metadata
for iExpt = 1:length(ExpRefs)
    ExpRef = {};
    try
        if isempty(dir(fullfile(ExpRefs{iExpt},'*2P*.tif')))
            raw_folders = dir(fullfile(ExpRefs{iExpt},'raw_imaging_data*'));
            for i = 1:length(raw_folders)
                ExpRef{i}= fullfile(ExpRefs{iExpt},raw_folders(i).name);
            end
        else
            ExpRef = ExpRefs(iExpt);
        end
        for i = 1:length(ExpRef)
            meta = mesoscopeMetadataExtraction(ExpRef{i},'alyx',alyx);
            fn_refs = {'reference','hirez'}; %subfolders that may contain reference images/stacks
            for iF = 1:length(fn_refs)
                if exist(fullfile(ExpRef{i},fn_refs{iF}),'dir')
                    mesoscopeMetadataExtraction_ref(fullfile(ExpRef{i},fn_refs{iF}),'alyx',alyx);
                end
            end
        end
    catch ME
        rethrow(ME)
        %warning(getReport(ME))
        %warning(sprintf('Something was wrong, skipping this session.\n'));
    end
end

%% 4) TODO copy to server

% Miles' copy script??
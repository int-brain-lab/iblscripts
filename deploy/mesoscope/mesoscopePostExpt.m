function out = mesoscopePostExpt(ExpRefs)

%this function is to be run after a mesoscope experiment at the scanimage PC.
%1) TODO looks for today's data
%2) writes Frame QC
%3) extracts metadata
%4) TODO copies data to server

%for testing
if nargin<1
    ExpRefs = {'Y:\Subjects\test\2023-03-03\002\raw_imaging_data_00'};
end

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

%% 1) TODO automatically search today's data


%% 2) write Frame QC
for iExpt = 1:length(ExpRefs)
    try
        ExpRef = ExpRefs{iExpt};
        writeFrameQC(ExpRef,'auto');
    catch ME
        rethrow(ME)
    end
end

%% 3) extract metadata
for iExpt = 1:length(ExpRefs)
    try
        ExpRef = ExpRefs{iExpt};
        meta = mesoscopeMetadataExtraction(ExpRef,'alyx',alyx);
    catch ME
        rethrow(ME)
    end
end

%% 4) TODO copy to server
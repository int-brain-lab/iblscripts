%this script is to be run after a mesoscope experiment at the scanimage PC.
%1) looks for today's data
%2) writes Frame QC
%3) extracts metadata (including 3D projection)
%4) copies data to server

ExpRefs = {...
    '2023-06-27_2_SP044',...
    '2023-06-27_3_SP044'};

%make sure we are logged into alyx
if exist('alyx') && isa(alyx,'Alyx')
    if ~alyx.IsLoggedIn
        alyx = alyx.login;
    end
else
    alyx = Alyx();
end

%% 1) TODO automatically search today's data


%% 2) write Frame QC
for iExpt = 1:length(ExpRefs)
    ExpRef = ExpRefs{iExpt};
    writeFrameQC(ExpRef,'auto');
end

%% 3) extract metadata
for iExpt = 1:length(ExpRefs)
    ExpRef = ExpRefs{iExpt};
    meta = mesoscopeMetadataExtraction(ExpRef,'alyx',alyx);
end

%% 4) TODO copy to server
%this script is to be run after a mesoscope experiment at the scanimage PC.
%1) looks for today's data
%2) writes Frame QC
%3) extracts metadata (including 3D projection)
%4) copies data to server

%input the expRefs to be processed (can be raw paths)
ExpRefs = {...
    ...%'Y:\Subjects\SP037\2023-02-08\001\raw_imaging_data_00',... %QC done, no meta yet
    ...%'Y:\Subjects\SP037\2023-02-09\001\raw_imaging_data_00',... %QC done, no meta yet
    ...%'Y:\Subjects\SP037\2023-02-10\001\raw_imaging_data_00',... %QC done, no meta yet
    ...%'Y:\Subjects\SP037\2023-02-14\001\raw_imaging_data_00',... %QC done
    ...%'Y:\Subjects\SP037\2023-02-16\001\raw_imaging_data_00',... %QC done
    ...%'Y:\Subjects\SP037\2023-02-23\001\raw_imaging_data_00',... %QC done
    ...%'Y:\Subjects\SP037\2023-02-23\001\raw_imaging_data_01',... %QC done
    ...%'Y:\Subjects\SP037\2023-02-24\001\raw_imaging_data_00',... %QC done
    ...%'Y:\Subjects\SP037\2023-02-24\001\raw_imaging_data_01',... %QC done
    ...%'Y:\Subjects\SP037\2023-03-09\001\raw_imaging_data_00',... %QC done
    ...%'Y:\Subjects\SP037\2023-03-09\001\raw_imaging_data_01',... %QC done
    ...%'Y:\Subjects\SP037\2023-03-23\002\raw_imaging_data_00',... %QC done
    ...%'Y:\Subjects\SP037\2023-03-23\002\raw_imaging_data_01',... %QC done
    ...%'Y:\Subjects\SP037\2023-03-24\001\raw_imaging_data_00',... %QC done
    ...%'Y:\Subjects\SP037\2023-03-24\001\raw_imaging_data_01',... %QC done
    ...%'Y:\Subjects\SP037\2023-03-27\001\raw_imaging_data_00',... %dual-plane recording (at same depth)
    ...%'Y:\Subjects\SP037\2023-03-27\001\raw_imaging_data_01',... %dual-plane recording (at same depth)
    'Y:\Subjects\SP037\2023-03-28\001\raw_imaging_data_00',...
    'Y:\Subjects\SP037\2023-03-28\001\raw_imaging_data_01',...
    'Y:\Subjects\SP037\2023-03-28\001\raw_imaging_data_02'...
    };

%for testing on one data-file
%ExpRefs = {'Y:\Subjects\SP037\2023-02-20\002\raw_imaging_data_00'};

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
%% Init
clear all
close all
clc
%% Session - path to data
filepath_NPY ='/Users/gaelle/Downloads/Alex_test/'; 
filepath_Metadata = '/Users/gaelle/Downloads/Alex_test/';

%% Read NPY 
synch_pol = ReturnDataNPY([filepath_NPY filesep '*_spikeglx_sync.polarities*']);
synch_tim = ReturnDataNPY([filepath_NPY filesep '*_spikeglx_sync.times*']);
synch_cha = ReturnDataNPY([filepath_NPY filesep '*_spikeglx_sync.channels*']);

%%  Hardware settings
%  -- Screen
Fscreen = 60; % Hz, Ipad screen refresh rate

% -- Map HW channels, see ibllib.io.extractors.ephys_fpga
%    TODO : Hardcoded, should be loaded from channel map metadata (if exists)
ch_frame2ttl = 12; % only io needed for this protocol

%% Read meta data
% Extracted from .raw BIN file
RFmetadata = ReturnMetadataRaw([filepath_Metadata filesep '*RFMapStim.raw*'],'BIN') ;

% Extracted from .raw JSON file
TaskSettings = ReturnMetadataRaw([filepath_Metadata filesep '*taskSettings.raw*'],'JSON') ;

%% Extract stimulus order and version from metadata
ID_stim = TaskSettings.STIM_ORDER;
version = TaskSettings.IBLRIG_VERSION_TAG;

% Replace . by space to convert into numerical values
version(strfind(version,'.')) = ' ';
version = str2num(version);

%% Init var

% -- Stim and Spacer
n_stim = length(ID_stim);
id_spacer = find(ID_stim==0);
n_expected_spacer = length(id_spacer) ;

% -- Screen
Tscreen = 1/Fscreen ;

%% Receptive field mapping - find expected TTLs
% -- Init var
Mgrid = size(TaskSettings.VISUAL_STIM_1.dva_mat,1);
Ngrid = size(TaskSettings.VISUAL_STIM_1.dva_mat,2);

% -- Reshape RFmetadata matrix (MxNxframes)
r_RFmetadata = reshape(RFmetadata, Mgrid, Ngrid, []);

pr_RFmetadata = permute(r_RFmetadata, [2, 1, 3]);

TTL_data = squeeze(pr_RFmetadata(1,1,:));

% -- Convert values to 0,1,-1 for simplicity
TTL_data01 = zeros(size(TTL_data));
TTL_data01(find(TTL_data==0)) = -1;
TTL_data01(find(TTL_data==255)) = 1;

% -- Find number of passage from [128 0] and [128 255]  (converted to 0,1,-1)
d_TTL_data01 = diff(TTL_data01);

id_raise = find(TTL_data01==0 & [d_TTL_data01; 0]==1);
id_fall = find(TTL_data01==0 & [d_TTL_data01; 0]==-1);

% -- number of rising TTL pulse expected in frame2ttl trace
TTL_data_Rise_Expected = ...   
    length(id_raise) + ...
    length(id_fall);


%% Init expected TTL for each stimulus
nTTL_Stim_expected = zeros(1,n_stim);

for i_stim = 1:n_stim     
    if ID_stim(i_stim) == 5 % spont. act.
       nTTL_Stim_expected(i_stim) = 0;
    elseif ID_stim(i_stim) ==1 % RF
        nTTL_Stim_expected(i_stim) = TTL_data_Rise_Expected;
    else
       eval(['nTTL_Stim_expected(i_stim) = TaskSettings.VISUAL_STIM_' num2str(ID_stim(i_stim)) '.ttl_num;']); 
    end
end
clear i_stim

%% Find time of rising pol (1)
polarity=1;
channelID = ch_frame2ttl;
[synch_Ris_tim] = find_PolTS_SynchCh(channelID,synch_cha,synch_tim,synch_pol,polarity);

polarity=-1;
[synch_Fal_tim] = find_PolTS_SynchCh(channelID,synch_cha,synch_tim,synch_pol,polarity);

clear polarity channelID

sort_RF = sort([synch_Fal_tim ; synch_Ris_tim]);

%% Find stimulus spacer timestamps
% --- Time between state (frame) change from Metadata
diff_frame_time = diff( Tscreen * TaskSettings.VISUAL_STIM_0.ttl_frame_nums );

% --- Build model of TTL response
jitter = Tscreen * 3; 
model_frame = jitter + diff_frame_time(1+2 : end-2); % remove 2 extreme values

% --- High values are zeroed
dRF = diff(sort_RF) ;
dRF(find(dRF > max(model_frame))) = 0;

% --- Convolve TTL raw trace against model
C = conv2(dRF,model_frame); plot(C,'.-b')

% --- TODO Hardcoded ; set threshold to find spacer
thrsh = 3;
indx_C = find(C(1:end-2)<thrsh & C(2:end-1)>thrsh &  C(3:end)<thrsh);

% --- Remove some data added because of convolution
crop_l = floor(length(model_frame)/2) ;

spacer_ts_middle = sort_RF(indx_C-crop_l+2) ; 
n_spacer = length(spacer_ts_middle);

if n_spacer ~= n_expected_spacer
    error('number of spacer invalid')
end

clear crop_l C indx_C thrsh dRF jitter


%% PLOT CHECK
% -- General
% figure
% plot(sort_RF(1:end-1), diff(sort_RF),'.-k')
% hold on; 
% plot(synch_Ris_tim,0.3,'.b'); 
% plot(synch_Fal_tim,0.3,'.r'); 
% line([0 synch_Ris_tim(end)],[0.2 0.2]) 
% ylim([0 2])


% -- Spacer
% figure;
% plot(sort_RF,1,'.k'); hold on
% plot(spacer_ts_middle,1,'r*')



%% Find frame2ttl pulse ID associated with each stim sequence and assign timestamps

% --- Add dead time around spacer middle timestamps
half_l_spacer = cumsum(diff_frame_time(1:ceil(length(diff_frame_time)/2))) ;
half_l_spacer = half_l_spacer(end);

spacer_end = half_l_spacer + 1; % TODO Hardcoded; 3 seconds added between spacer and any stimulus in protocol

% --- Init
cell_ts = cell(1,n_stim); 

version_check = version>=[5,2,6];
if ~ismember(0, version_check) 
    
    for i_stim = 1:n_stim
        if ~ismember(i_stim,id_spacer)
                % check number of ttl found in between parser
                id_ts_stim = find(...
                    sort_RF > spacer_ts_middle(i_stim/2) + spacer_end & ...
                    sort_RF < spacer_ts_middle(i_stim/2+1) - spacer_end) ;
                
                if ID_stim(i_stim)== 1
                    cell_ts{i_stim} = sort_RF(id_ts_stim(2:2:end-1));  % *2 for rise/fall , + 1 as Bonsai generates 1 pulse at begining and end
                elseif ID_stim(i_stim) == 2
                    cell_ts{i_stim} = sort_RF(id_ts_stim(3:end)); % +2 at beginning, rapid transient artifact due to Bonsai loading
                else
                    cell_ts{i_stim} = sort_RF(id_ts_stim);
                end
                
                if length(cell_ts{i_stim}) ~= nTTL_Stim_expected(i_stim)
                    % plot(sort_RF(id_ts_stim),0.35,'.g')
                    error('TTL number wrong')
                end
        
        end
    end
    
else
    error('case by case extractor needed')   
end

%% Assign further timestamps for RF mapping

i_stim = find(ID_stim == 1); % Todo change, hardcoded and assumes there is only 1 presentation of RF mapping

X = sort([id_raise; id_fall]);% TTLs ID
T = cell_ts{i_stim};
nframe = size(pr_RFmetadata,3);
Xq = [1:nframe];
Tq = interp1(X,T,Xq);

% Note: interpolation returns (NAN) at borders, so doing crude linear
% spacing

% first values
Tq(1:X(1)-1) = Tq(X(1))-fliplr([1:1:X(1)-1])*Tscreen;
% end values
Tq(X(end)+1:end) = Tq(X(end))+[1:1:length(Tq(X(end)+1:end))]*Tscreen;


%% output cell containing timestamps for each stimulus

cell_ts_output = cell_ts;
cell_ts_output{i_stim} = Tq ; % replace so as to have a timestamps at each frame for RF mapping



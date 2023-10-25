function paths = prepareExperiment(expRef, stub, varargin)
% prepareExperiment Save an experiment description stub for a given session
%   Saves a stub _ibl_experiment.description.yaml file for use by the
%   iblrig data transfer script.
%
%   Inputs (Positional):
%     expRef (char) - An experiment reference string.
%     stub (char) - Either 'timeline' or 'mesoscope', or a path to a yaml
%       file to load.
%
%   Inputs (Optional Name-Value Parameters):
%     computerID (char) - The name of the computer to keep track of which
%       PC created which stub file.
%     fullPathInSettings (logical) - If false, adds Subjects/ to settings
%       paths.
%
%   Outputs:
%     paths (cell) - The full path(s) of the saved stub file.
%
%   Example 1: Save a stub on the mesoscope computer
%     paths = prepareExperiment('2023-01-01_1_SP035', 'mesoscope')
%
%   Example 2: Save with a custom stub file and computer ID
%     prepareExperiment('2023-01-01_1_SP035', 'C:\stub.yaml', 'computerID', 'syncPC');

% Ensure the yaml toolbox is installed
assert(~isempty(which('yaml.dumpFile')),...
  'Rigbox:setup:toolboxRequired',...
  ['Requires yaml toolbox. '...
   'Click <a href="matlab:web(''%s'',''-browser'')">here</a> to'...
   ' install.'],...
   'https://uk.mathworks.com/matlabcentral/fileexchange/106765-yaml')

% Generate a default unique label for this device using the computer
% hostname and volume label
[ret, name] = system('hostname');
assert(ret==0)
[ret, out] = dos('vol');
assert(ret==0)
sc = strsplit(out, '\n');
volLbl = sc{2}(end-8:end);
transfer_id = [strip(name) '_' volLbl];

% Parse the input args
p = inputParser;
s = '(?<date>^\d{4}-\d\d\-\d\d)_(?<seq>\d+)_(?<subject>\w+)';  % dat.expRefRegExp
isExpRef = @(r) (ischar(r) || isstring(r)) && ~isempty(regexp(r, s, 'once'));
addRequired(p, 'expRef', isExpRef)
addRequired(p, 'stub', @ischar)
addOptional(p, 'computerID', transfer_id, @ischar)
addOptional(p, 'fullPathInSettings', true, @islogical)
p.parse(expRef, varargin{:})


% Load the paths from the iblrig settings
settingsPath = 'C:\iblrig\settings\iblrig_settings.yaml';
if ~exist(settingsPath, 'file')
  error('%s does not exist: please set up iblrigv8', settingsPath)
end
settings = yaml.loadFile(settingsPath);
if ~(isfield(settings, 'iblrig_local_data_path') && ...
    isstring(settings.iblrig_local_data_path))
  error('%s not setup correctly: missing or invalid iblrig_local_data_path', settingsPath)
end
% Parse the local and remote paths
localPath = settings.iblrig_local_data_path;
if ~p.Results.fullPathInSettings
  % Behave like iblrig add append <lab>/Subjects/ to remote path
  if isfield(settings, 'ALYX_LAB') && ~yaml.isNull(settings.ALYX_LAB)
    localPath = fullfile(localPath, settings.ALYX_LAB);
  end
  localPath = fullfile(localPath, 'Subjects');
end
if isfield(settings, 'iblrig_remote_data_path') && ...
    isstring(settings.iblrig_remote_data_path)
  remotePath = settings.iblrig_remote_data_path;
  if ~p.Results.fullPathInSettings
    % Behave like iblrig add append Subjects/ to remote path
    remotePath = fullfile(remotePath, 'Subjects');
  end
else
  remotePath = false;
end
if remotePath == false
  warning('No remote path set up, will only save locally')
end

% Parse the expRef
parsed = regexp(expRef, s, 'names');
expSequence = cellstr2double(parsed.seq);  % ensure not zero padded

% Create or load stub
switch stub
  case 'mesoscope'
    stub = struct('devices', ...
      struct('mesoscope', ...
        struct('mesoscope', ...
          struct('collection', 'raw_imaging_data*', 'sync_label', 'chrono'))), ...
      'procedures', 'Imaging',...
      'version', '1.0.0');
  case 'timeline'
    stub = struct('sync', ...
      struct('nidq', ...
        struct('acquisition_software', 'timeline', 'collection', 'raw_sync_data', 'extension', 'npy')),...
      'version', '1.0.0');
  otherwise
    % Assumes a path to a stub file
    if exist(stub, 'file') == 0
      error('stub file "%s" not found', stub)
    end
    stub = yaml.loadFile(stub);
end

% Save local
localSession = fullfile(localPath, parsed.subject, parsed.date, sprintf('%d', expSequence));
localFile = fullfile(localSession, ['_ibl_experiment.description_' p.Results.computerID '.yaml']);
if exist(localSession, 'dir') == 0
  warning('%s does not exists, creating folder(s)', localSession)
  status = mkdir(localSession);
  assert(status == 1, 'Failed to create remote session folder(s) %s', localSession)
end
yaml.dumpFile(localFile, stub, 'block')
assert(exist(localFile, 'file') ~= 0, 'Failed to save stub to file %s', localFile)
fprintf('Saved stub to %s\n', localFile);
paths = {localFile};

% Save remote
if remotePath ~= false
  remoteSession = fullfile(remotePath, parsed.subject, parsed.date, sprintf('%03d', expSequence));
  remoteFile = fullfile(remoteSession, '_devices/', [expRef '@' p.Results.computerID '.yaml']);
  if exist(remoteSession, 'dir') == 0
    warning('%s does not exists, creating folder(s)', remoteSession)
    status = mkdir(remoteSession);
    assert(status == 1, 'Failed to create remote session folder(s) %s', remoteSession)
  end
  yaml.dumpFile(remoteFile, stub, 'block')
  assert(exist(remoteFile, 'file') ~= 0, 'Failed to save stub to file %s', remoteFile)
  fprintf('Saved stub to %s\n', remoteFile);
  paths = [paths {remoteFile}];
end

end

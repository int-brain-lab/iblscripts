
function processIBLUDP(src, evt)
% PROCESSIBL Parse messages from IBL rig Communicator
%   Expects msg to be an JSON array where the first element is an
%   integer corresponding to an io.ExpSignal enumeration.
%
%   All unsolicited messages are immediately echo'd as confirmation of
%   receipt. When an additional message is sent in response, we expect
%   the remote host to echo it, however this is not essential.
%
% See also io.ExpSignal
persistent lastSent % the last sent message for processing echoes
listenerName = 'IBLUDP';

ipstr=src.DatagramAddress;
port=src.DatagramPort;

src.RemoteHost = ipstr;
src.RemotePort = port;

data=fread(src, src.BytesAvailable);
msg = char(data');
% tstr  = datestr(now, 'YYYY-mmm-DD HH:MM:SS.fff');
% fprintf('[%s %s] Received ''%s'' from %s:%d\n', listenerName, tstr, msg, ipstr, port);

% Check if message is an echo of a previously sent message
if isequal(msg, lastSent)
    log('%s: confirmation received from %s:%i', listenerName, ipstr, port);
    return % do nothing
else
    log('%s: ''%s'' received from %s:%i', listenerName, msg, ipstr, port);
    % Echo as confirmation of receipt
    fwrite(src, data);
end

% parse the message
% Example parsed message: {[20];'2022-01-01_1_subject';[]}
try
    info = jsondecode(msg);
    signal = info{1};
    data = info(2:end);
    signal = io.ExpSignal(signal);
catch err  % Failed to parse message
    respond(0, err.message, msg)
end

log('%s: processing %s signal', signal);
switch signal
    case 'EXPINIT'
        % Experiment is initializing.
        r = struct('status', int8(getStatus()));
        %         r.exp_ref = iff(tlObj.IsRunning, tls.expRef, data.exp_ref);
        respond(signal, r) % nothing to do; just send initialized signal
    case 'EXPSTART'
        % iblrig starts and experiment

        status = getStatus(); % get local status
        % and report back with current ExpRef
        respond(signal, src.UserData.app.ExpRef, struct('status', int8(status)))

    case {'EXPEND', 'EXPINTERRUPT', 'EXPCLEANUP'}
        % Experiment has stopped on iblrig
        respond(signal, int8(getStatus())) 
    case 'EXPSTATUS'
        % Experiment status.
        respond(signal, int8(getStatus()))
    case 'EXPINFO'
        % Experiment info, including task protocol start and end.
        status = getStatus();
        r = struct('main_sync', true, 'api_version',  '1.0.0', 'status', int8(status));
        r.exp_ref = src.UserData.app.ExpRef;
        if ~strcmp(data{2}.subject, dat.parseExpRef(r.exp_ref))
            warning('Remote behaviour PC has incorrect subject selected')
        end
        respond(signal, int8(status), r)
    case 'ALYX'
        % Alyx token.
        % TODO Handle case where base_url is sent alone
        % if numel(rmEmpty(data)) == 0
        %     % Send token
        %     ai = tls.AlyxInstance;
        %     if ai.IsLoggedIn
        %         token = struct(ai.User, struct('token', struct(ai).Token));
        %         respond(signal, ai.BaseURL, token)
        %     else
        %         respond(signal, NaN, struct)
        %     end
        % else
        %     % Install token
        %     [base_url, token] = data{:};
        %     user = first(fieldnames(token));
        %     aiObj = struct(...
        %         'BaseURL', base_url,...
        %         'Token', token.(user).token,...
        %         'User', user,...
        %         'QueueDir', Alyx('','').QueueDir,...
        %         'SessionURL', Alyx('','').SessionURL...
        %         );
        %     ai = Alyx.loadobj(aiObj);
        %     assert(ai.IsLoggedIn)
        %     tls.AlyxInstance = ai;
        % end
    otherwise
        warning('Unknown signal "%s"', signal)
end

    function respond(signal, varargin)
        % RESPOND Send a message to the remote host
        %   Send a JSON response to remote host. The message is assigned to
        %   lastSent so that the echo will be recognised as such.
        %
        %   For respond('EXPSTART', 2022-01-01_1_subject, NaN) the JSON
        %   message would be: '[20, "2022-01-01_1_subject", null]'.
        if signal ~= 0
            signal = io.ExpSignal(signal); % validate
        end
        response = jsonencode([{int8(signal)}, varargin]);
        fwrite(src, uint8(response'))
        lastSent = response;
        log('%s: sent %s to %s:%i', listenerName, response, ipstr, port);
    end

    function status = getStatus()
        % GETSTATUS Get the experiment status from Timeline
        hasExpRef = ~isempty(src.UserData.app.ExpRef);
        status = iff(hasExpRef, io.ExpStatus.RUNNING, io.ExpStatus.CONNECTED);
    end


end % of the main function


function log(varargin)
    message = sprintf(varargin{:});
    timestamp = datestr(now, 'dd-mm-yyyy HH:MM:SS');
    fprintf('[%s] %s\n', timestamp, message);
end



 

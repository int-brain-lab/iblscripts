function tls = testMesoUDP()

% mpepSendPort = 1103; % send responses back to this remote port
quitKey = KbName('esc');
manualStartKey = KbName('t');

%% Start UDP communication
listeners = struct(...
  'socket',...
    {pnet('udpsocket', 11001)},...           %ibl listening socket
  'callback',...
    {@processIBL},... % IBL message handling function
  'name', {'ibl'});
log('Bound UDP sockets');

tls.close = @closeConns;
tls.process = @process;
tls.listen = @listen;
tls.AlyxInstance = Alyx('','');


%% Helper functions

  function closeConns()
    log('Unbinding UDP socket');
    arrayfun(@(l) pnet(l.socket, 'close'), listeners);
  end

  function process()
    %% Process each socket listener in turn
    arrayfun(@processListener, listeners);
  end

  function processListener(listener)
    sz = pnet(listener.socket, 'readpacket', 1000, 'noblock');
    if sz > 0
      msg = pnet(listener.socket, 'read');
      listener.callback(listener, msg); % call special handling function
    end
  end


  function processIBL(listener, msg)
    % PROCESSIBL Parse messages from IBL rig Communicator
    %   Expects msg to be an JSON array where the first element is an
    %   integer corresponding to an io.ExpSignal enumeration.
    %
    %   All unsolicited messages are immediately echo'd as confirmation of
    %   receipt. When an additional message is sent in response, we expect
    %   the remote host to echo it, however this is not essential.
    %
    % See also io.ExpSignal

    persistent lastSent % the last send message for processing echoes
   
    function respond(signal, varargin)
      % RESPOND Send a message to the remote host
      %   Send a JSON response to remote host. The message is assigned to
      %   lastSent so that the echo will be recognised as such.
      %
      %   For respond('EXPSTART', 2022-01-01_1_subject, NaN) the JSON
      %   message would be: '[20, "2022-01-01_1_subject", null]'.
      signal = io.ExpSignal(signal); % validate
      response = jsonencode([{int8(signal)}, varargin]);
      pnet(listener.socket, 'write', response);
      lastSent = response;
      pnet(listener.socket, 'writepacket', ipstr, port);
      log('%s: sent %s to %s:%i', listener.name, response, ipstr, port);
    end
    
    function status = getStatus()
      % GETSTATUS Get the experiment status from Timeline
      hasRef = ~isempty(pick(tls, 'expRef', 'def', []));
      status = iff(hasRef, io.ExpStatus.RUNNING, io.ExpStatus.CONNECTED);
    end

 
    % Retrieve remote host IP for logging
    [ip, port] = pnet(listener.socket, 'gethost');
    ip = num2cell(ip);
    ipstr = sprintf('%i.%i.%i.%i', ip{:});
    
    % Check if message is an echo of a previously sent message
    if isequal(msg, lastSent)
      log('%s: confirmation received from %s:%i', listener.name, ipstr, port);
      return % do nothing
    end
    
    log('%s: ''%s'' from %s:%i', listener.name, msg, ipstr, port);
    % Echo as confirmation of receipt
    pnet(listener.socket, 'write', msg);
    pnet(listener.socket, 'writepacket', ipstr, port);
    
    % parse the message
    % Example parsed message: {[20];'2022-01-01_1_subject';[]}
    try
      info = jsondecode(msg);
      signal = info{1};
      data = info(2:end);
      signal = io.ExpSignal(signal);
    catch err  % Failed to parse message
      respond(0, err.message, signal)
    end
      
    switch signal
      case 'EXPINIT'
        % Experiment is initializing.
        r = struct('status', int8(getStatus()));
%         r.exp_ref = iff(tlObj.IsRunning, tls.expRef, data.exp_ref);
        respond(signal, r) % nothing to do; just send initialized signal
      case 'EXPSTART'
        % Experiment has begun.
        if tlObj.IsRunning
          status = getStatus();
          respond(signal, tls.expRef, struct('status', int8(status)))
        else
          try
%             tls.expRef = data{1};
%             tlObj.start(tls.expRef, tls.AlyxInstance);
%             assert(tlObj.IsRunning)
            status = getStatus();
            respond(signal, tls.expRef, struct('status', int8(status))) % Let server know we've started
          catch err
            respond(io.ExpSignal.EXPINTERRUPT, err)
          end
        end
      case {'EXPEND', 'EXPINTERRUPT'}
        % Experiment has stopped or interrupt received.
%         tlObj.stop();
%         assert(~tlObj.IsRunning)
        respond(signal, int8(getStatus())) % Let server know we've stopped
      case 'EXPCLEANUP'
        % Experiment cleanup begun.
        status = getStatus();
        respond(signal, int8(status))  % Do nothing
      case 'EXPSTATUS'
        % Experiment status.
        respond(signal, int8(getStatus()))
      case 'EXPINFO'
        % Experiment info, including task protocol start and end.
        status = getStatus();
        r = struct('main_sync', true, 'api_version',  '1.0.0', 'status', int8(status));
        r.exp_ref = tls.expRef;
        if ~strcmp(data{1}.subject, dat.parseExpRef(r.exp_ref))
          warning('Remote behaviour PC has incorrect subject selected')
        end
        respond(signal, int8(status), r)
      case 'ALYX'
        % Alyx token.
        % TODO Handle case where base_url is sent alone
        if numel(rmEmpty(data)) == 0
          % Send token
          ai = tls.AlyxInstance;
          if ai.IsLoggedIn
            token = struct(ai.User, struct('token', struct(ai).Token));
            respond(signal, ai.BaseURL, token)
          else
            respond(signal, NaN, struct)
          end
        else
          % Install token
          [base_url, token] = data{:};
          user = first(fieldnames(token));
          aiObj = struct(...
            'BaseURL', base_url,...
            'Token', token.(user).token,...
            'User', user,...
            'QueueDir', Alyx('','').QueueDir,...
            'SessionURL', Alyx('','').SessionURL...
          );
          ai = Alyx.loadobj(aiObj);
          assert(ai.IsLoggedIn)
          tls.AlyxInstance = ai;
        end
      otherwise
        warning('Unknown signal "%s"', signal)
    end
  end

  function listen()
    % polls for UDP instructions for starting/stopping timeline
    % listen to keyboard events
    KbQueueCreate();
    KbQueueStart();
    cleanup1 = onCleanup(@KbQueueRelease);
    log(['Polling for UDP messages. PRESS <%s> TO QUIT, '...
      '<%s> to manually start/stop timeline'],...
      KbName(quitKey), KbName(manualStartKey));
    running = true;
    tls.expRef = '2024-06-12_1_iblrig_test_subject';
    tid = tic;
    while running
      process();
      [~, firstPress] = KbQueueCheck;
      if firstPress(quitKey)
        running = false;
      end
      if toc(tid) > 0.2
        pause(1e-3); % allow timeline aquisition every so often
        tid = tic;
      end
    end
  end

  function log(varargin)
    message = sprintf(varargin{:});
    timestamp = datestr(now, 'dd-mm-yyyy HH:MM:SS');
    fprintf('[%s] %s\n', timestamp, message);
  end

end

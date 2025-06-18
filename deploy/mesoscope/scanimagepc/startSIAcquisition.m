function startSIAcquisition(ExpRef)

% will start ScanImage acquisition
% useful whent the ExpStart UDP was missed for whatever reason or when
% you want to start an acquisition manually with a pre-defined ExpRef (to
% take care of all the filenames etc.)

[subject, date, seq] = dat.expRefToMpep(ExpRef);
msg = sprintf('ExpStart %s %d %d', subject, date, seq);

u = udp(myIP);
fopen(u);
u.RemotePort = 1001; % this should be the port number SIMesoListener is listening on 
fwrite(u, msg);
fclose(u);
delete(u);
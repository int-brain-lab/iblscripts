function stopSIAcquisition()

% will stop ScanImage acquisition
% useful whent the ExpEnd UDP was missed for whatever reason or when
% you just need to stop acquisition manually and get ready for the next one

fakeExpRef = '1900-01-01_1_Forced';
[subject, date, seq] = dat.expRefToMpep(fakeExpRef);
msg = sprintf('ExpEnd %s %d %d', subject, date, seq);

u = udp(myIP);
fopen(u);
u.RemotePort = 1001; % this should be the port number SIMesoListener is listening on
fwrite(u, msg);
fclose(u);
delete(u);
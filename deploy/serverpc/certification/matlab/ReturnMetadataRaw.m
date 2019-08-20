function [outvar] = ReturnMetadataRaw(filename,type)
% filename contains path to file and file name (containing *)
f1 = dir(filename);
fname = [f1.folder filesep f1.name];
switch type
    case 'BIN'
        fileID = fopen(fname,'r');
        outvar = fread(fileID) ;

    case 'JSON'
        outvar = jsondecode(fileread(fname));
end
end


function [outvar] = ReturnDataNPY(filename)
% filename contains path to file and file name
f1 = dir(filename); 
outvar = io.read.npy([f1.folder filesep f1.name]);
end


function data = readTiffFastNoSkip(fileName)

if nargin<1
    [fn, fp] = uigetfile('G:\Temp2p\*.*');
    fileName = fullfile(fp, fn);
end

fInfo = imfinfo(fileName);
nPixels = fInfo(1).Width * fInfo(1).Height;
bytesPerImage = nPixels * fInfo(1).BitsPerSample/8;
firstOffset = fInfo(1).Offset;
interFrameOffset = diff([fInfo(1:2).Offset])-bytesPerImage;
headerSamples = interFrameOffset/fInfo(1).BitsPerSample*8;
fid = fopen(fileName, 'r');
fseek(fid, firstOffset, 'bof');
data = fread(fid, '*int16');
fclose(fid);
data = reshape(data, headerSamples+nPixels, []);
data = data(headerSamples+1:end, :);
% the data in tiffs is written row-wise, so the first dimension is width
data = reshape(data, fInfo(1).Width, fInfo(1).Height, []);
% and then we need to transpose the frames
data = permute(data, [2, 1, 3]);

function data = readTiffFast(fileName, options, fInfo)

if nargin<2 || isempty(options)
    options.firstFrame = 1;
    options.lastFrame = Inf;
    options.frameStride = 1; % useful for reading only a specific channel/plane
end
if nargin<3
    fInfo = imfinfo(fileName);
end

nPixels = fInfo(1).Width * fInfo(1).Height;
bytesPerImage = nPixels * fInfo(1).BitsPerSample/8;
firstOffset = fInfo(1).Offset;
interFrameOffset = diff([fInfo(1:2).Offset])-bytesPerImage;
fid = fopen(fileName, 'r');
%     tic
if any(diff(diff([fInfo.Offset])))
    % this is the complicated case, where headers are different in size
    % I don't actually have any data to properly test/debug this part...
    headerBytes = diff([fInfo.Offset]);
    headerBytes(end+1) = fInfo(end).FileSize - fInfo(end).Offset;
    headerBytes = headerBytes - bytesPerImage;
    headerSamples = headerBytes/fInfo(1).BitsPerSample*8;
    % search for the first frame (including the header)
    fseek(fid, firstOffset, 'bof');
    % read the whole file
    data = fread(fid, '*int16');
    % throw away all the headers, leave only images
    pixIdx = false(size(data));
    % figure out which samples to keep
    startIdx = cumsum(headerSamples+nPixels)-nPixels+1;
    endIdx = cumsum(headerSamples+nPixels);
    for iFrame = options.firstFrame:options.frameStride:length(fInfo)
        pixIdx(startIdx(iFrame):endIdx(iFrame)) = true;
    end
    % only keep the image data
    data = data(pixIdx);
else
    % this is the simple case, where all the headers are the same size
    % search for the beginning of the first image (also skip the first header)
    fseek(fid, fInfo(options.firstFrame).Offset + interFrameOffset, 'bof');
    % read only image data, skipping the headers
    bytesPerFullFrame = diff([fInfo(1:2).Offset]);
    extraSkipBytes = (options.frameStride - 1) * bytesPerFullFrame;
    data = fread(fid, [num2str(nPixels) '*int16=>int16'], interFrameOffset+extraSkipBytes);
end
%     toc
fclose(fid);
% the data in tiffs is written row-wise, so the first dimension is width
data = reshape(data, fInfo(1).Width, fInfo(1).Height, []);
% and then we need to transpose the frames
data = permute(data, [2, 1, 3]);

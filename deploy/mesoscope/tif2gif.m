function out = tif2gif(filename, options)

if nargin<2 || isempty(options)
    options.firstFrame = 1;
    options.lastFrame = Inf;
    options.frameStride = 1; % useful for reading only a specific channel/plane
end

fr_gif = 10; %gif frame rate in Hz
[ff,fn,fext] = fileparts(filename);
savefolder = fullfile(ff,'..','snapshots');
if ~exist(savefolder,'dir'), mkdir(savefolder); end

fprintf('Loading images... ');
rawData = readTiffFast(filename,options); %this is a faster tiff loading function
fprintf('Done!\n');

fprintf('Saving gif in %s... ',savefolder);
savename = [fn '.gif']; % Specify the output file name
%savename = 'referenceStack.gif'; % Specify the output file name
[minval,maxval] = bounds(rawData(:));
minval = double(minval);
%maxval = double(maxval);
maxval = double(prctile(rawData(:),99.9));
for idx = 1:size(rawData,3)
    %[A,map] = rgb2ind(squeeze(rawData(:,:,idx)),256);
    A = uint8(((double(squeeze(rawData(:,:,idx)))-minval)/(maxval-minval))*255);
    if idx == 1
        imwrite(A,fullfile(savefolder,savename),'gif','LoopCount',Inf,'DelayTime',1/fr_gif);
    else
        imwrite(A,colormap(gray),fullfile(savefolder,savename),'gif','WriteMode','append','DelayTime',1/fr_gif);
    end
end
fprintf('Done!\n');
end
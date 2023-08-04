% location and names of relevant data files
dataFolder = '~/Documents/PYTHON/iblscripts/deploy/mesoscope';
annotationsFile = 'annotation_volume_10um_by_index.npy';
% NOTE: order of dimensions is [AP, DV, ML]
% use this flag to show intermediate results of triangulations etc.
doPlotting = false;
% this on is unnecessary for the task
% tv = readNPY(fullfile(dataFolder, dataFile));
% load annotation data
av = readNPY(fullfile(dataFolder, annotationsFile));
% av==1 is annotation of outside of the brain
binary = av~=1;
% what are the dimensions of the data arrays
[nAP, nDV, nML] = size(av);
axisAP = 0:nAP-1;
axisDV = 0:nDV-1;
axisML = 0:nML-1;
% Let's shift the coordinated realtive to bregma, and also start working in
% millimeters (might want to scale to micromenters for consistency with ephys?)
voxelSize = 10; % [um] resolution of the atlas
% [540 0 570] in Andy's case (in voxel number)
bregmaCoords = [5400 332 5739] / voxelSize; % IBL bregma coordinate

axisAPum = (axisAP - bregmaCoords(1)) * voxelSize * -1;
axisDVum = (axisDV - bregmaCoords(2)) * voxelSize * -1;
axisMLum = (axisML - bregmaCoords(3)) * voxelSize;
%% finding the top brain voxels on the dorsal surface
surfaceDV = nan(nAP, nML);
for iAP = 1:nAP
    for iML = 1:nML
      depth = find(binary(iAP, :, iML), 1, 'first');
      if ~isempty(depth)  
          surfaceDV(iAP, iML) = axisDVum(depth);
      end
    end
end
if doPlotting
    hSurfaceFig = figure;
    s = surf(axisMLum, axisAPum, surfaceDV);
    s.LineStyle = 'None';
    axis equal tight
    colorbar
    xlabel('ML [um]')
    ylabel('AP [um]')
    hold on;
end
%% Calculating the convex hull of the brain and making a triangulation representaion of it
% find the borders of the brain in 3D, this is pretty straighforward on a binary image
binaryEdges = edge3(binary, 'sobel', 0.5, 'nothinning');
[idxAP, idxDV, idxML] = ind2sub(size(binaryEdges), find(binaryEdges(:)));
% transform to millimeters
AP = axisAPum(idxAP)';
DV = axisDVum(idxDV)';
ML = axisMLum(idxML)';
% find a convex hull of the brain boundary (this will take 'a few seconds' to compute - 
% about two minutes on my desktop)
% [AP, DV, ML] is an nPoints-by-3 array with points in 3D we want to find
% the convex hull of
tic
k = convhull([ML, AP, DV], 'Simplify', true);
toc
% k is nVertices-by-3 array with triplets of points from [ML, AP, DV] that
% from triangles of the convex hull. Not all points from [ML, AP, DV]
% participate in convex hull (sulci), so the next step is to remove unnecessary data
% Only keep the points that participate in the convex hull representation
% This will help keep the triangulation representaion memory footprint smaller
kUnique = unique(k(:)); % check which points are used at least ones
% recalculation is very quick (no junk points)
k = convhull([ML(kUnique), AP(kUnique), DV(kUnique)], 'Simplify', true);
% the size of k did not change (number of rows is the number of triangles), but we
% only need to save a small subset of [ML, AP, DV] points
fullTR = triangulation(k, [ML(kUnique), AP(kUnique), DV(kUnique)]);
% fullTR = 
% 
%   triangulation with properties:
% 
%               Points: [8438×3 double] - these are our [ML. AP. DV] from before
%     ConnectivityList: [16872×3 double] - this is our k from before
%% Only keep faces that have normals pointing up (positive DV value)
% Here we use the property of the convex hull as it is calculated/represented by MATLAB
% if you use faceNormal(TR object) it will give a normal vector pointing
% out of the convex hull
% If we port this to Python, we will need to check if this holds, otherwise
% things might get a bit more complicated
points = fullTR.Points;
normals = faceNormal(fullTR);
upFaces = find(normals(:, 3)>0);
kNew = k(upFaces, :); % only keep triangles that have normal vector with positive DV component
dorsalTR = triangulation(kNew, fullTR.Points); % ideally should also remove unused vertices, but this is a minro point
normals = faceNormal(dorsalTR);
if doPlotting
    figure(hSurfaceFig);
    hold on;
    t = trimesh(dorsalTR);
    t.FaceAlpha = 0;
    t.EdgeColor = [0 0 0];
    axis equal tight
    faceIncenters = incenter(dorsalTR);
    hold on;
    plot3(faceIncenters(:, 1), faceIncenters(:, 2), faceIncenters(:, 3), '.r')
    % this seems to be working very well
end
%% Create a flat two dimensional triangulation (dorsal projection)
% this will be used later to assign each point in ML-AP to a specific face
% in the dorsalTR
% we just basically remove the third coordinate, at that's it
flatTR = triangulation(dorsalTR.ConnectivityList, [dorsalTR.Points(:, 1), dorsalTR.Points(:, 2)]);
if doPlotting
    hFlatProjection = figure;
    triplot(flatTR);
    axis equal;
    xlabel('ML [um]');
    ylabel('AP [um]');
end
%% Save the necessary things for later use
st.axisMLum = axisMLum; % ML coordinates in um, corresponding to 1..nML points in the atlas volume
st.axisAPum = axisAPum; % the same for AP
st.axisDVum = axisDVum; % the same for DV
st.fullTR = fullTR; % triangulation object of the whole convex hull of the brain volume
st.dorsalTR = dorsalTR; % only contains triangles that point 'up' - dorsal surface
st.flatTR = flatTR; % 2-D projection of the dorsalTR (simply removed the DV coordinate)
st.surfaceDV = surfaceDV; % DV coordinates of the dorsal surface of the brain
st.voxelSize = voxelSize; % atlas voxel size in um
save('mlapdvAtlas_test.mat', '-struct', 'st')

% Save the connectivity list and points of the full triangulation as numpy files.
% These were cast to ints and resaved into the surface_triangulation.npz file.
writeNPY(fullTR.ConnectivityList - 1, 'fullTR.connectivityList_ibl.npy')  % index from 0
writeNPY(fullTR.Points, 'fullTR.points_ibl.npy')
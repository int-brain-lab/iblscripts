# File fixtures
### fullTR
A triangle mesh of the smoothed convex hull of the dorsal surface of the mouse brain, generated from
the 2017 Allen 10um annotation volume.

- **fullTR.connectivityList.npy** - An N by 3 integer array of vertex indices defining all points that form a triangle.
- **fullTR.points.npy** - An N by 3 float array of x-y vertices, defining all points of the triangle mesh.

### dorsalTR
The triangle mesh with only the faces that have normals pointing up (i.e. a positive DV value).

- **dorsalTR.connectivityList.npy** - An N by 3 integer array of vertex indices defining all points that form a triangle.
- **dorsalTR.points.npy** - An N by 3 float array of x-y vertices, defining all points of the triangle mesh.

### flatTR
A 2-D projection of the dorsalTR (simply removed the DV coordinate).

- **flatTR.connectivityList.npy** - An N by 3 integer array of vertex indices defining all points that form a triangle.
- **flatTR.points.npy** - An N by 3 float array of x-y vertices, defining all points of the triangle mesh.


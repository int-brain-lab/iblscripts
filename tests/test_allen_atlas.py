import numpy as np

from ibllib.atlas import AllenAtlas
self = AllenAtlas(25)

# extracts the top surface from the volume and make sure it's all populated
ix, iy = np.meshgrid(np.arange(self.bc.nx), np.arange(self.bc.ny))
iz = self.bc.z2i(self.top)
inds = self._lookup_inds(np.stack((ix, iy, iz), axis=-1))
assert np.all(self.label.flat[inds][~np.isnan(self.top)] != 0)
# one sample above, it's all zeros
inds = self._lookup_inds(np.stack((ix, iy, np.maximum(iz - 1, 0)), axis=-1))
assert np.all(self.label.flat[inds][~np.isnan(self.top)] == 0)
# plt.imshow(self._label2rgb(self.label.flat[inds]))  # show the surface

# do the same for the bottom surface
izb = self.bc.z2i(self.bottom)
inds = self._lookup_inds(np.stack((ix, iy, izb), axis=-1))
assert np.all(self.label.flat[inds][~np.isnan(self.top)] != 0)
# one sample below, it's all zeros
inds = self._lookup_inds(np.stack((ix, iy, np.maximum(izb + 1, 0)), axis=-1))
assert np.all(self.label.flat[inds][~np.isnan(self.bottom)] == 0)


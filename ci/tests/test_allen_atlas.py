import numpy as np
import matplotlib.pyplot as plt
from ibllib.atlas import AllenAtlas, FlatMap, BrainRegions
from ibllib.atlas.flatmaps import plot_swanson


def test_simple():
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


def test_flatmaps():
    fm = FlatMap(flatmap='dorsal_cortex')
    fm.plot_flatmap(depth=0)

    fm = FlatMap(flatmap='circles')
    fm.plot_flatmap()

    fm = FlatMap(flatmap='pyramid')
    fm.plot_flatmap(volume='image')


def test_swanson():
    br = BrainRegions()

    # prepare array of acronyms
    acronyms = ['ACAd1', 'ACAv1', 'AId1', 'AIp1', 'AIv1', 'AUDd1', 'AUDp1', 'AUDpo1', 'AUDv1',
                'SSp-m1', 'SSp-n1', 'SSp-tr1', 'SSp-ul1', 'SSp-un1', 'SSs1',
                'VISC1', 'VISa1', 'VISal1', 'VISam1', 'VISl1', 'VISli1', 'VISp1', 'VISp2/3', 'VISpl1', 'VISpm1',
                'SSp-n2/3', 'SSp-tr2/3', 'SSp-ul2/3', 'SSp-un2/3', 'SSs2/3',
                'VISC2/3', 'VISa2/3', 'VISal2/3', 'VISam2/3', 'VISl2/3', 'VISli2/3', 'VISp2/3', 'VISpl1', 'VISpl2/3']

    # assign data to each acronym
    values = np.arange(len(acronyms))

    regions_rl = np.r_[br.acronym2id(acronyms), -br.acronym2id(acronyms)]
    values_rl = np.random.randn(regions_rl.size)

    fig, ax = plt.subplots()
    plot_swanson(hemisphere='both', br=br)
    fig, ax = plt.subplots()
    plot_swanson(br=br)
    fig, ax = plt.subplots()
    plot_swanson(acronyms, values, cmap='Blues', br=br)
    fig, ax = plt.subplots()
    plot_swanson(acronyms, values, cmap='Blues', hemisphere='both')
    fig, ax = plt.subplots()
    plot_swanson(regions_rl, values_rl, hemisphere='both', cmap='magma', br=br)

    plt.close('all')

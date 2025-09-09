import unittest
import unittest.mock
import tempfile
import pickle
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from ScanImageTiffReader import ScanImageTiffReader
from iblatlas.atlas import AllenAtlas
from one.alf.path import ALFPath
import one.alf.io as alfio
from one.api import ONE

from ibllib.io.extractors import mesoscope
from ibllib.mpci import registration

from ci.tests.base import IntegrationTest

_logger = logging.getLogger('ibllib')


class TestMesoscopeRegistration(IntegrationTest):

    required_files = ['mesoscope/SP058/2024-08-01/001/raw_imaging_data_02/reference/referenceImage.stack.tif']

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def test_register_reference_stacks(self):
        """Test the registration of reference stacks."""
        target_stack_path = Path(self.data_path, self.required_files[0])
        # Rotate and offet for testing purposes
        stack = ScanImageTiffReader(str(target_stack_path)).data()
        transform = skimage.transform.EuclideanTransform(rotation=1, translation=(1, 2))
        stack = skimage.transform.warp(stack, transform, output_shape=stack.shape)
        # Save the transformed stack to a temporary file
        stack_path = self.tmp_path / 'referenceImage.stack.tif'
        # stack_path.parent.mkdir(parents=True, exist_ok=True)
        # ScanImageTiffReader.write(stack_path, stack)
        import tifffile
        tifffile.imwrite(stack_path, stack, photometric='rgb')

        # Load the reference stacks
        aligned, params = registration.register_reference_stacks(stack_path, target_stack_path)
        # register_reference_stacks(stack_path, target_stack_path, crop_size=390, apply_threshold=False, display=False, **kwargs)


class TestReferenceSession(unittest.TestCase):
    """Test extraction of FOV coordinates for aligned reference session."""

    def setUp(self):
        self.one = ONE()
        # self.session_path = ALFPath(r'D:\Flatiron\alyx.internationalbrainlab.org\cortexlab\Subjects\SP037\2024-08-01\001')
        self.reference_session = '839bb5b1-120f-49d0-b7c9-5174c0c66b5a'  # SP037/2023-02-20/001
        self.session_path = self.one.eid2path(self.reference_session)
        self.reprojection = registration.MesoscopeFOVHistology(self.session_path, one=self.one)
        self.reprojection.reference_session = self.reference_session
        self.reprojection.get_signatures()
        self.reprojection.data_handler = self.reprojection.get_data_handler()
        # Download required datasets
        dsets = self.one.list_datasets(self.session_path)
        required = ['mpciROIs.stackPos.npy', 'experiment.description.yaml',
                    'referenceImage.meta.json', 'referenceImage.stack.tif',
                    '_ibl_rawImagingData.meta.json']
        dsets = [d for d in dsets if any(d.endswith(r) for r in required)]
        # self.one.load_datasets(self.session_path, dsets, download_only=True)  # commented out because of size mismatches


    def test_get_brain_surface_plane_from_ref_points(self):
        """This tests that the output exactly matches Georg's original code for this session."""
        self.reprojection.atlas = AllenAtlas(res_um=25)
        reference_image = self.reprojection._load_reference_stack()
        p_ref, n_ref, dv_avg = self.reprojection.get_brain_surface_plane_from_ref_points(reference_image)
        expected_p_ref = np.array([0.003011, -0.001025, -0.000125])
        expected_n_ref = np.array([7.34976797e-04, 1.20536195e-01, 9.92708661e-01])
        expected_dv_avg = 150.0
        np.testing.assert_array_almost_equal(p_ref, expected_p_ref, decimal=5)
        np.testing.assert_array_almost_equal(n_ref, expected_n_ref, decimal=5)
        self.assertAlmostEqual(dv_avg, expected_dv_avg, delta=1e-2)

    def test_ref_session(self):
        # 1. Download the MLAPDV coordinates of the reference session
        # registered_mlapdv = self.reprojection.get_atlas_registered_reference_mlap(self.reference_session)
        # ref_mlapdv = np.load(registered_mlapdv)
        # mlapdv = self.reprojection.interpolate_FOVs()
        # # 2.
        # get_rois_mlapdv_rel
        # rois_mlapdv_from_rel
        with unittest.mock.patch.object(self.reprojection, 'update_craniotomy_center'):
            self.reprojection._run()

    def test_ref_session_with_save(self):
        # 1. Download the MLAPDV coordinates of the reference session
        # registered_mlapdv = self.reprojection.get_atlas_registered_reference_mlap(self.reference_session)
        # ref_mlapdv = np.load(registered_mlapdv)
        # mlapdv = self.reprojection.interpolate_FOVs()
        # # 2.
        # get_rois_mlapdv_rel
        # rois_mlapdv_from_rel
        self.reprojection.atlas = AllenAtlas(res_um=25)
        # Load the reference stack & (down)load the registered MLAPDV coordinates
        reference_image = self.reprojection._load_reference_stack()
        # Load main meta
        _, meta_files, _ = self.reprojection.input_files[0].find_files(self.reprojection.session_path)
        meta = mesoscope.patch_imaging_meta(alfio.load_file_content(meta_files[0]) or {})
        nFOV = len(meta.get('FOV', []))

        with open(self.session_path / 'interpolated_fovs_rel.pkl', 'rb') as f:
            mlapdv_rel = pickle.load(f)

        # Account for optical plane tilt
        if (self.session_path / 'mlapdv_final_georg.pkl').exists():
            with open(self.session_path / 'mlapdv_final_georg.pkl', 'rb') as f:
                fovs = pickle.load(f)
        else:
            fovs = dict.fromkeys(range(nFOV), None)

        # mlapdv_rel = self.correct_fov_depth_and_surface_projection(mlapdv, meta, reference_image)
        done = sum(v is not None for v in fovs.values())
        _logger.info('%i/%i processed', done, nFOV)
        if done == nFOV:
            return
        i = next(i for i in fovs if fovs[i] is None)
        _logger.info('Processing FOV %i', i)

        mean_image_mlapdv = self.reprojection.project_mlapdv_from_surface(mlapdv_rel[i:i+1])
        fovs[i] = mean_image_mlapdv[0]
        with open(self.session_path / 'mlapdv_final_georg.pkl', 'wb') as f:
            pickle.dump(fovs, f)

    def test_project_mlapdv_from_surface_georg(self):
        """This tests that the output exactly matches Georg's original code for this session."""
        self.reprojection.atlas = AllenAtlas(res_um=25)
        # Load test points from file
        with open(self.session_path / 'interpolated_fovs_rel.pkl', 'rb') as f:
            file_result = pickle.load(f)
        fov, idx = np.unique(file_result[0].reshape(-1, 3), axis=0, return_index=True)
        # Load expected results from file
        with open(self.session_path / 'mlapdv_final_georg_simplified.pkl', 'rb') as f:
            expected_result = pickle.load(f)
        # Run the method
        result = self.reprojection.project_mlapdv_from_surface([fov])
        # Compare results
        expected = expected_result[0].reshape(-1, 3)[idx]
        np.testing.assert_array_almost_equal(result[0], expected, decimal=5)

    def test_save_roi(self):
        with open(self.session_path / 'mlapdv_final_georg.pkl', 'rb') as f:
            fovs = pickle.load(f)

        with unittest.mock.patch.object(self.reprojection, 'update_craniotomy_center'), \
                unittest.mock.patch.object(self.reprojection, 'interpolate_FOVs'), \
                unittest.mock.patch.object(self.reprojection, 'correct_fov_depth_and_surface_projection'), \
                unittest.mock.patch.object(self.reprojection, 'project_mlapdv_from_surface', return_value=list(fovs.values())):
            self.reprojection._run()


class TestSession(unittest.TestCase):
    """Test extraction of FOV coordinates for non-reference session."""

    def setUp(self):
        self.one = ONE()
        self.session_path = ALFPath(r'D:\Flatiron\alyx.internationalbrainlab.org\cortexlab\Subjects\SP037\2023-03-09\001')
        self.reference_session = '839bb5b1-120f-49d0-b7c9-5174c0c66b5a'  # SP037/2023-02-20/001
        # Download required datasets
        dsets = self.one.list_datasets(self.session_path)
        required = ['mpciROIs.stackPos.npy', 'experiment.description.yaml',
                    'referenceImage.meta.json', 'referenceImage.stack.tif',
                    '_ibl_rawImagingData.meta.json']
        dsets = [d for d in dsets if any(d.endswith(r) for r in required)]
        # self.one.load_datasets(self.session_path, dsets, download_only=True)  # commented out because of size mismatches
        self.reprojection = registration.MesoscopeFOVHistology(self.session_path, one=self.one)
        self.reprojection.reference_session = self.reference_session
        self.reprojection.get_signatures()
        self.reprojection.data_handler = self.reprojection.get_data_handler()

    def test_session_with_save(self):
        # 1. Download the MLAPDV coordinates of the reference session
        # registered_mlapdv = self.reprojection.get_atlas_registered_reference_mlap(self.reference_session)
        # ref_mlapdv = np.load(registered_mlapdv)
        # mlapdv = self.reprojection.interpolate_FOVs()
        # # 2.
        # get_rois_mlapdv_rel
        # rois_mlapdv_from_rel
        self.reprojection.atlas = AllenAtlas(res_um=25)
        # Load the reference stack & (down)load the registered MLAPDV coordinates
        reference_image = self.reprojection._load_reference_stack()
        # Load main meta
        _, meta_files, _ = self.reprojection.input_files[0].find_files(self.reprojection.session_path)
        meta = mesoscope.patch_imaging_meta(alfio.load_file_content(meta_files[0]) or {})
        nFOV = len(meta.get('FOV', []))

        f = self.session_path / 'interpolated_fovs.pkl'
        if f.exists():
            with open(f, 'rb') as f:
                mlapdv = pickle.load(f)
        else:
            mlapdv = self.reprojection.interpolate_FOVs(reference_image, meta)
            with open(f, 'wb') as f:
                pickle.dump(mlapdv, f)

        f = self.session_path / 'interpolated_fovs_rel.pkl'
        if f.exists():
            with open(f, 'rb') as f:
                mlapdv_rel = pickle.load(f)
        else:
            mlapdv_rel = self.reprojection.correct_fov_depth_and_surface_projection(mlapdv, meta, reference_image)
            with open(f, 'wb') as f:
                pickle.dump(mlapdv_rel, f)

        # Account for optical plane tilt
        if (self.session_path / 'mlapdv_final_georg.pkl').exists():
            with open(self.session_path / 'mlapdv_final_georg.pkl', 'rb') as f:
                fovs = pickle.load(f)
        else:
            fovs = dict.fromkeys(range(nFOV), None)

        # mlapdv_rel = self.correct_fov_depth_and_surface_projection(mlapdv, meta, reference_image)
        done = sum(v is not None for v in fovs.values())
        _logger.info('%i/%i processed', done, nFOV)
        if done == nFOV:
            return
        i = next(i for i in fovs if fovs[i] is None)
        _logger.info('Processing FOV %i', i)

        # mean_image_mlapdv = self.reprojection.project_mlapdv_from_surface(mlapdv_rel[i:i+1])
        # fovs[i] = mean_image_mlapdv[0]
        mean_image_mlapdv = self.reprojection.project_mlapdv_from_surface(mlapdv_rel)
        fovs = {i: x for i, x in enumerate(mean_image_mlapdv)}
        with open(self.session_path / 'mlapdv_final_georg.pkl', 'wb') as f:
            pickle.dump(fovs, f)


r"""
     191688/262144 [05:50<01:28, 794.84it/s]
Projecting MLAPDV points 6/6:  73%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                           | 191760/262144 [05:50<02:08, 547.65it/s]
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "c:\Users\Work\Documents\github\ibllib-repo\ibllib\mpci\registration.py", line 1669, in project_mlapdv_from_surface
    p, n_vec = get_plane_at_point_mlap(point[0], point[1], vertices, connectivity_list)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Work\Documents\github\ibllib-repo\ibllib\mpci\brain_meshes.py", line 87, in get_plane_at_point_mlap
    face, ix = get_closest_face(faces, ln0)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "c:\Users\Work\Documents\iblenv\Lib\site-packages\numba\np\old_arraymath.py", line 615, in array_argmin_impl_float
    raise ValueError("attempt to get argmin of an empty sequence")
ValueError: attempt to get argmin of an empty sequence
"""
# Step 1
# If there is a histology image, load that as our reference stack MLAP coordinates

# If the image is from a different session, calculate the transformation between the two
# and apply that to the reference stack

# Re-calculate center MM coordinates by using the MLAP coordinates of the centre pixel,
# then add the offset? TODO Confirm whether we need to use offset. I guess it's cleaner to
# use the craniotomy centre for computing the normal vector

# If there is no histology image, compute the estimate coordinates from the convex hull
# of the atlas and the craniotomy coordinates

# Step 2
# Used


# p_ref, n_ref, dv_avg = get_brain_surface_plane_from_ref_points(
#     ref_surface_points, ref_img_meta
# )


class FOVRefactor(registration.MesoscopeFOVHistology):
    def reproject(self, points, display=False):
        """TODO Document and rename.

        This method will estimate position based on a plane drawn from experimenter defined points at the brain's surface.
        For this a JSON of selected points is required as well as the reference image and its meta file.

        Parameters
        ----------
        points : dict
            A dictionary of points selected by the experimenter.  Expected format:
            {'points': [{'stack_idx': int, 'coords': [int, int]}, ...], 'range': [int, int]}.

        Assumptions: all on the same X plane, all the same width
        """
        stack, ref_meta = self._load_reference_stack()
        xy_res = np.array([
            ref_meta['rawScanImageMeta']['XResolution'],
            ref_meta['rawScanImageMeta']['YResolution']
        ])
        if ref_meta['rawScanImageMeta']['ResolutionUnit'].casefold() == 'centimeter':
            # NB: these values are (x, y) in μm and shouldn't be used with mlap coordinates without rotation
            px_per_um = xy_res * 1e-4
            um_per_px = 1 / px_per_um
        else:
            raise NotImplementedError('Reference image resolution unit must be in centimeters')
        # these can not be used because they seem to refer to the raw acquisition and not the stack
        # ref_meta["rawScanImageMeta"]["Height"], ref_meta["rawScanImageMeta"]["Width"]

        # need to get the image shape from the .stack and not the .raw
        ref_stack_n_px = np.flip(np.array(stack.shape[1:]))  # in (x, y)
        # known: the point in the image in pixel coordinates and mlap space
        center = ref_meta['centerMM']
        # An array of the positive ML and AP directions in (x, y) pixel space
        rotation_matrix = np.c_[ref_meta['imageOrientation']['positiveML'],
                                ref_meta['imageOrientation']['positiveAP']]
        craniotomy_center_offset = np.array([center['x'], center['y']]) * 1e3  # μm from center
        mlap = np.array([center['ML'], center['AP']]) * 1e3  # μm from bregma

        # Where is bregma in pixel space?
        # ml is along the y axis, and is positive, so longditudinal fissure is at a large +ve y value
        # ref_stack_n_px[1]/2 = 270 px, which is mlap[0] = 2600 μm from bregma, so absolute bregma ml
        # distance is mlap[0] * px_per_um[1] = 260 so bregma is at 260 + 270 = 530 px
        #
        # ap is along the x axis, and is negative, so the back of the brain is at a large -ve x value
        # ref_stack_n_px[0]/2 = 245 px, which is mlap[1] = -2000 μm from bregma, so absolute bregma ap
        # distance is mlap[1] * px_per_um[0] = 200 so bregma is at 245 - 200  = 45 px

        def px2um(px):
            """Map pixel (x, y) coordinates to MLAP coordinates."""
            # Calculate the pixel offset from the image center
            image_center_px = ref_stack_n_px / 2
            craniotomy_pixel = image_center_px + (craniotomy_center_offset / um_per_px)
            pixel_offset = px - craniotomy_pixel  # origin now the craniotomy pixel

            # Apply the scaling factor to convert pixel distances to mlap units
            pixel_offset_um = pixel_offset * um_per_px

            # Rotate the pixel offset using the orientation vector
            inv_rotation_matrix = np.linalg.inv(rotation_matrix)
            rotated_offset = np.dot(pixel_offset_um, inv_rotation_matrix)

            # Translate the rotated coordinates to the mlap space using the craniotomy coordinates
            mlap_coords = mlap + rotated_offset

            return mlap_coords

        def um2px(um):
            """Maps mlap coordinates to pixel space."""
            # Calculate the mlap offset from the craniotomy coordinates
            mlap_offset = np.array(um) - np.array(mlap)
            # Rotate the mlap offset using the orientation vector
            rotated_offset_xy = np.dot(mlap_offset, rotation_matrix)

            # Apply the scaling factor to convert mlap distances to pixel units
            rotated_offset_px = rotated_offset_xy / um_per_px

            # Translate the rotated coordinates to the pixel space using the craniotomy pixel coordinates
            image_center_px = ref_stack_n_px / 2
            craniotomy_pixel = image_center_px + (craniotomy_center_offset / um_per_px)
            return craniotomy_pixel + rotated_offset_px


        # Sanity checks
        bregma_px = np.array([45, 530])  # (x, y) coordinates of bregma in pixel space
        craniotomy_px = um2px(mlap)
        np.testing.assert_array_equal(um2px(mlap), ref_stack_n_px / 2)
        np.testing.assert_array_equal(px2um(craniotomy_px), mlap)
        np.testing.assert_array_equal(np.round(um2px([0, 0])).astype(int), bregma_px)
        np.testing.assert_array_equal(px2um(um2px([0, 0])), [0, 0])

        # Check works with multiple points
        bregma_um = np.zeros((3, 2))
        np.testing.assert_array_equal(np.round(um2px(bregma_um)), np.tile(bregma_px, (3, 1)))
        np.testing.assert_array_equal(px2um(np.tile(craniotomy_px, (3, 1))), np.tile(mlap, (3, 1)))

        # TODO All px
        x1, x2 = np.meshgrid(np.arange(ref_stack_n_px[0]), np.arange(ref_stack_n_px[1]))
        xy_coords = np.array((x1, x2)).T.reshape(-1, 2)
        px2um(xy_coords)

        if display:  # pragma: no cover
            for i, point in enumerate(points['points']):
                # Convert the point to pixels
                x, y = np.array(point['coords']) * ref_stack_n_px
                fig, ax = plt.subplots()
                ax.matshow(stack[point['stack_idx'], :, :], cmap='gray')
                ax.set_xlabel('n px'), ax.set_ylabel('n px')
                ax.xaxis.set_label_position('top')
                ax.plot([x - 24, x + 24], [y, y], lw=2, alpha=0.7, color='r')
                ax.plot([x, x], [y - 24, y + 24], lw=2, alpha=0.7, color='r')
                landmark_coords_um = px2um([x, y])
                ax.annotate(f'landmark ({landmark_coords_um[0]:.4g}, {landmark_coords_um[1]:.4g})',
                            (x, y), color='r', xytext=(10, -10), textcoords='offset points')
                ax.xaxis.tick_top()

                # Plot bregma
                ax.axhline(um2px([0, 0])[1], lw=1, color='blue', alpha=0.5)
                ax.axvline(um2px([0, 0])[0], lw=1, color='blue', alpha=0.5)
                ax.annotate('bregma (0, 0)', um2px([0, 0]), color='b', xytext=(10, -10), textcoords='offset points')

                image_center_px = ref_stack_n_px / 2
                craniotomy_pixel = image_center_px - (craniotomy_center_offset / um_per_px)

                ax.plot(craniotomy_pixel[0], craniotomy_pixel[1], 'go', alpha=0.7, markersize=12, markerfacecolor='none')
                ax.annotate(f'craniotomy ({mlap[0]:.4g}, {mlap[1]:.4g})',
                            craniotomy_px, color='g', xytext=(10, -10), textcoords='offset points')
                # Sadly, secondary axes are not supported for matshow
                # # secax_x = ax.secondary_xaxis('bottom', functions=(px2um, um2px))
                # # secax_x.set_xlabel(('ML' if positive_mlap[0][0] else 'AP') + ' / um')
                # # secax_x.xaxis.set_tick_params(rotation=70
                # # secax_y = ax.secondary_yaxis('right', functions=(px2um, um2px))
                # # secax_y.set_ylabel(('ML' if positive_mlap[0][1] else 'AP') + ' / um')
                # # secax_y.yaxis.set_tick_params(rotation=70)

                fig.canvas.manager.set_window_title(f'Point {i} - stack #{point["stack_idx"]}')
            plt.show()

    def mlapdv_from_rel(
        self,
        mlapdv_rel: np.ndarray,
        atlas_res: int = 25,
    ):
        """
        now we have corrected ml ap coordinates of cells in the imaging plane
        we first need to determine where those cells are on the surface of the atlas
        and from that point, move down along the local brain normal by the true dv

        project onto the atlas either along the brain normal of the atlas, or the adjusted brain
        normal as calculated from the reference points. I think the atlas normal makes more sense
        (as the influence of the difference between the two angles has been accounted for) but
        left in here optionally to test.

        Args:
            mlapdv_rel (np.ndarray): rois_mlapdv as expressed in the imaging plane. from previous step
            ref_img_meta (dict): _description_
            ref_surface_points (dict): _description_
            project_along_reference (bool, optional): . Defaults to False.

        Returns:
            _type_: _description_

        NB: This method failed to reproduce Georg's results and should be removed.
        """
        # atlas = MRITorontoAtlas(atlas_res)
        atlas = AllenAtlas(res_um=25)  # 25 μm resolution
        atlas.compute_surface()
        # vertices, connectivity_list = calculate_surface_triangulation(atlas)  # Doesn't work for some reason
        # Load triangulation in μm
        vertices, connectivity_list = self.load_triangulation(atlas=atlas)  # atlas=atlas

        _logger.info(f'Min-max ML vertex: {vertices[:, 0].min():.6f} to {vertices[:, 0].max():.6f} meters')
        _logger.info(f'Min-max AP vertex: {vertices[:, 1].min():.6f} to {vertices[:, 1].max():.6f} meters')
        _logger.info(f'Min-max DV vertex: {vertices[:, 2].min():.6f} to {vertices[:, 2].max():.6f} meters')

        mlapdv = []
        mlapdv_surface = []
        # Pre-calculate triangulation data for optimization
        # Cache triangle equations (plane coefficients) for all triangles
        for fov in tqdm(mlapdv_rel):
            fov_flat = fov.reshape(-1, 3) * 1e6  # fov is in m, here we convert toum. Now both triangulation and fov values are in μm
            fov_mlapdv = np.empty_like(fov_flat)
            fov_mlapdv_surface = np.empty_like(fov_flat)

            # Vectorized triangle finding
            mlap_points = fov_flat[:, :2]

            # Find triangles for all points at once (more efficient than loop)
            face_indices = np.fromiter(
                (find_triangle(mlap, vertices[:, :2], connectivity_list.astype(np.intp)) for mlap in mlap_points),
                dtype=np.intp
            )

            # Group points by triangle for batch processing
            unique_faces = np.unique(face_indices)

            for face_idx in unique_faces:
                # Get all points that belong to this triangle
                point_mask = face_indices == face_idx
                point_indices = np.where(point_mask)[0]

                if len(point_indices) == 0:
                    continue

                # Get triangle vertices and calculate normal once per triangle
                face_vertices = vertices[connectivity_list[face_idx, :], :]
                n = surface_normal(face_vertices)

                # Ensure normal points deeper into brain (positive DV direction)
                # Since DV increases with depth, normal should point in +DV direction
                if n[2] < 0:
                    n *= -1
                # TODO cache
                abc, *_ = np.linalg.lstsq(face_vertices, np.ones(3), rcond=None)

                # Vectorized surface point calculation for all points in this triangle
                mlap_batch = fov_flat[point_indices, :2]
                coord_dv_batch = (1 - mlap_batch @ abc[:2]) / abc[2]
                surface_points = np.column_stack([mlap_batch, coord_dv_batch])

                # Apply depths vectorized
                depths = fov_flat[point_indices, 2]

                # Debug: Check depth values
                _logger.info(f"Triangle {face_idx}: depths range {depths.min():.6f} to {depths.max():.6f} meters")
                _logger.info(f"Normal vector: {n}")

                final_points = surface_points + np.outer(depths, n)
                final_points_ = np.array([p + n * -1 * d for p, d in zip(surface_points, depths)])
                # final_points = surface_points + n * -1 * depths
                # fov_mlapdv[i] = p + n * -1 * fov_flat[i, 2]  # <- the true depth

                # Store results
                fov_mlapdv_surface[point_indices] = surface_points / 1e6
                fov_mlapdv[point_indices] = final_points / 1e6

            mlapdv.append(fov_mlapdv.reshape(*fov.shape))
            mlapdv_surface.append(fov_mlapdv_surface.reshape(*fov.shape))
        import pickle
        with open(self.session_path / 'mlapdv_final.pkl', 'wb') as f:
            pickle.dump([mlapdv, mlapdv_surface], f)

        #### PLOTS ####
        # Old surf
        # _vertices, _ = self.load_triangulation(legacy=True)
        # axes = plt.figure(figsize=[10, 10]).add_subplot(projection="3d")
        # axes.plot_trisurf(_vertices[:, 0], _vertices[:, 1], _vertices[:, 2], linewidth=0.2, antialiased=True)
        # New surf
        axes = plt.figure(figsize=[10, 10]).add_subplot(projection="3d")
        axes.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], linewidth=0.2, antialiased=True, alpha=0.5)
        # Plot optical axis plane (optical axis is 3x3 array of three 3-D vertices)
        optical_axis, normal = self._optical_axis_plane
        axes.plot_trisurf(optical_axis[:, 0], optical_axis[:, 1], optical_axis[:, 2], linewidth=0.2, antialiased=True)
        # Plot the optical axis as a much larger plane so we can see where it intersects with the surface

        # Create a large rectangular plane extended from the optical axis plane
        # Use the center point of the optical axis triangle and the normal vector
        center_point = np.mean(optical_axis, axis=0)
        # Define the size of the large rectangular plane
        plane_size = 0.01  # 10mm extension in each direction
        # Create two orthogonal vectors in the plane
        # First, find a vector that's not parallel to the normal
        if abs(normal[0]) < 0.9:
            v1 = np.cross(normal, [1, 0, 0])
        else:
            v1 = np.cross(normal, [0, 1, 0])
        v1 = v1 / np.linalg.norm(v1)  # normalize
        # Second orthogonal vector
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)  # normalize
        # Create the four corners of the rectangular plane
        large_plane_corners = np.array([
            center_point - plane_size * v1 - plane_size * v2,
            center_point + plane_size * v1 - plane_size * v2,
            center_point + plane_size * v1 + plane_size * v2,
            center_point - plane_size * v1 + plane_size * v2
        ])
        # Plot the large rectangular plane
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        axes.add_collection3d(Poly3DCollection([large_plane_corners], alpha=0.3, linewidths=1,
                                              edgecolors='red', facecolors='yellow', label='Extended optical axis plane'))

        # plot brain surface points
        from ibllib.mpci.brain_meshes import get_surface_points
        from ibllib.mpci.plotters import plot_brain_surface_points
        brain_surface_points = get_surface_points(atlas)
        axes = plot_brain_surface_points(brain_surface_points, ds=4, axes=axes)
        # Plot orginal MLAPDV points (but only the ROIs) as 3D scatter
        for i, fov in enumerate(mlapdv_rel):
            yx_pos = alfio.load_file_content(self.session_path / f'alf/FOV_{i:02}' / 'mpciROIs.stackPos.npy')
            roi_mlapdv = fov[yx_pos[:, 0], yx_pos[:, 1], :]
            axes.scatter(*roi_mlapdv.T, ".", c="b", s=1, alpha=0.05, label='Original MLAPDV points')
        # Plot the triangles that were used
        unique_faces = np.unique(face_indices)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        for face in unique_faces:
            face_vertices = vertices[connectivity_list[face, :], :]
            axes.add_collection3d(Poly3DCollection([face_vertices], alpha=.25, linewidths=1, edgecolors='r'))

        # Plot ROIs
        for i, fov in enumerate(mlapdv):
            # Plot the points in mlapdv space
            yx_pos = alfio.load_file_content(self.session_path / f'alf/FOV_{i:02}' / 'mpciROIs.stackPos.npy')
            roi_mlapdv = fov[yx_pos[:, 0], yx_pos[:, 1], :] * 1e6
            axes.scatter(*roi_mlapdv.T, ".", c="k", s=1, alpha=0.05, label='ROIs in imaging plane')
            # roi_surf = mlapdv_surface[i][yx_pos[:, 0], yx_pos[:, 1], :] * 1e6
            # axes.scatter(*roi_surf.T, ".", c="r", s=1, alpha=0.05, label='ROIs on brain surface')

        # Load Georg's
        import pickle
        with open(self.session_path / 'mlapdv_final_georg_simplified.pkl', 'rb') as f:
            processed = pickle.load(f)
        for i, fov in enumerate(processed):
            yx_pos = alfio.load_file_content(self.session_path / f'alf/FOV_{i:02}' / 'mpciROIs.stackPos.npy')
            roi_mlapdv = fov[yx_pos[:, 0], yx_pos[:, 1], :] * 1e6
            axes.scatter(*roi_mlapdv.T, ".", c="r", s=1, alpha=0.05, label='ROIs in imaging plane')

        return mlapdv
        # brain_surface_points = get_surface_points(atlas)

        # get the brain normal from ml, ap
        # FIXME Use updated center coordinates

        # center_mlap = np.array([ref_img_meta["centerMM"][d] for d in ["ML", "AP"]]) * 1e3
        # center_mlapdv, brain_normal = get_plane_at_point_mlap(
        #    *center_mlap, vertices, connectivity_list, numba=True
        # )
        # # create a new tilted coordinate system at the imaged plane
        # cs3d = setup_coordinate_systems_3d(center_mlapdv, brain_normal)

        # get the roi mlapdv values in the atlas space
        # again, just the incorrectly termed ml,ap coordinates. Setting DV to zero because the imaging plane is flat
        # and the points are just used for projection onto the brain surface

        # mlap0 = np.copy(mlapdv_rel)

        # mlap0[:, 2] = 0
        # _rois_mlapdv = cs3d.transform(mlap0 - center_mlapdv, "imaging_plane", "mlapdv")

        # project the rois onto the brain surface along the brain normal
        # adjusted for the sessions tilt
        # FIXME Skip this for histology resolved data!
        # rois_on_surface = np.zeros_like(_rois_mlapdv)
        # for i, roi in enumerate(tqdm(_rois_mlapdv)):
        #     faces, ips, ix = intersect_line_mesh_nb(
        #         vertices,
        #         connectivity_list,
        #         roi,
        #         brain_normal * -1,  # discuss if brain_normal or brain_normal_ref
        #     )
        #     face, ix = get_closest_face(faces, roi)
        #     rois_on_surface[i] = ips[ix]

        # and now go inward along the local brain normal, with true depth
        # this is the step that should be sensitive to the atlas resolution
        # as the local brain normal will differ more from ROI to ROI
        # rois_mlapdv = np.zeros_like(rois_on_surface)
        # for i, point in enumerate(tqdm(rois_on_surface)):
        #     p, n = get_plane_at_point_mlap(
        #         point[0], point[1], vertices, connectivity_list, numba=True
        #     )
        #     rois_mlapdv[i] = p + (n * -1 * mlapdv_rel[i, 2])  # <- the true depth

        # return (  # these are just now returned for plotting purposes
        #     rois_mlapdv,
        #     rois_on_surface,
        #     _rois_mlapdv,
        #     center_mlapdv,
        #     brain_normal,
        #     atlas,
        #     cs3d,
        # )


if __name__ == '__main__':
    suite = unittest.TestSuite()
    # suite.addTest(TestReprojectionUnit("test_interpolate_FOVs"))
    # suite.addTest(TestReprojection('test_load_mlapdv'))
    # suite.addTest(TestReferenceSession('test_save_roi'))
    # suite.addTest(TestReferenceSession('test_project_mlapdv_from_surface_georg'))
    suite.addTest(TestSession('test_session_with_save'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
    exit()


import tempfile
from pathlib import Path

import skimage.transform
from ScanImageTiffReader import ScanImageTiffReader

from ibllib.mpci import registration

from ci.tests.base import IntegrationTest

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

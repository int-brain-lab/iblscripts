import argparse
from pathlib import Path
import os

import numpy as np
from localmd.dataset import PMDDataset


class MotionBinDataset(PMDDataset):
    """Load a suite2p data.bin imaging registration file."""

    def __init__(self, dataset_path, ops_path):
        """
        Load a suite2p data.bin imaging registration file.

        Parameters
        ----------
        session_path : str, pathlib.Path
            The session path containing preprocessed data
        """
        self.dataset_path = Path(dataset_path)
        self.ops_path = Path(ops_path)
        self._shape = None
        
    @property
    def shape(self):
        """
        This property should return the shape of the dataset, in the form: (d1, d2, T) where d1
        and d2 are the field of view dimensions and T is the number of frames.

        Returns
        -------
        (int, int, int)
            The number of y pixels, number of x pixels, number of frames.
        """
        if self._shape is None:
            self._shape = self._compute_shape()
        return self._shape

    def _compute_shape(self):
        """
        Loads the suite2p ops file to retrieve the dimensions of the data.bin file.

        Returns
        -------
        (int, int, int)
            The number of y pixels, number of x pixels, number of frames.
        """
        ops = np.load(self.ops_path, allow_pickle=True).item()
        return ops['Ly'], ops['Lx'], ops['nframes']

    def get_frames(self, frames):
        """
        This function should take as input a list of integer-valued indices which describe the
        frames which need to be obtained from the dataset.

        These indices should be given in python-indexing convention (i.e. an index with value of 0
        refers to frame 1 of the dataset).

        Parameters
        ----------
        frames : list of int
            The frame numbers to retrieve (zero indexed).

        Returns
        -------
        numpy.array
            The frame data with shape (n y pixels, n x pixels, n frames).
        """
        file = np.memmap(self.dataset_path, mode='r', dtype=np.int16, shape=(self.shape[-1], *self.shape[:2]))
        try:
            frame_data = file[frames]
        finally:
            file._mmap.close()
        return np.transpose(frame_data, (1, 2, 0)).astype(np.float32)
    


if __name__ == '__main__':
    from localmd.decomposition import localmd_decomposition
    from localmd.visualization import generate_PMD_comparison_triptych

    import tifffile  # for saving comparison

    parser = argparse.ArgumentParser(
        prog='PMD', description='Run penalized matrix decomposition (PMD) on suite2p file registration bin file.')
    parser.add_argument('--dataset_path', type=str, help='The absolute path of the dataset wherever this script is being run')
    parser.add_argument('--ops_file', type=str, help='The absolute path of the ops file wherever this script is being run')
    parser.add_argument('--output_location', type=str, help='The location where the outputs should be saved on the filesystem where this is being run')
    # parser.add_argument('--fov', type=int, help=f'The field of view number')
    parser.add_argument('--block_height', default=20, type=int,
                        help='The height of the blocks in pixels (should be comparable to size of somata)')
    parser.add_argument('--block_width', default=20, type=int,
                        help='The width of the blocks in pixels (should be comparable to size of somata)')
    parser.add_argument('--frames_to_init', default=5000, type=int,
                        help='We begin the method by finding a low-rank (conservative) estimate of the linear subspace in which the signal resides.'
                             'We use frames_to_init frames, sampled at time points throughout the movie to compute this spatial basis.')
    parser.add_argument('--background_rank', default=1, type=int)
    parser.add_argument('--max_consec_failures', default=1, type=int)
    parser.add_argument('--max_components', default=40, type=int)
    

    pmd_params_dict = vars(parser.parse_args())
    dataset_path = pmd_params_dict.pop('dataset_path')
    ops_file_path = pmd_params_dict.pop('ops_file')
    output_folder=pmd_params_dict.pop('output_location')
    # FOV = pmd_params_dict.pop('fov')
    # SESSION_PATH = str(Path(pmd_params_dict.pop('session')).relative_to(ROOT))

    # For testing, check there are no bad frames
    # bad_frames_file = ROOT / SESSION_PATH / 'alf' / f'FOV_{FOV:02}' / 'mpci.badFrames.npy'
    # if not bad_frames_file.parent.exists():
    #     raise FileNotFoundError(str(bad_frames_file.parent))
    # assert not np.load(bad_frames_file).any(), 'bad frames in session'

    # current_dataset = MotionBinDataset(ROOT / SESSION_PATH)
    current_dataset = MotionBinDataset(dataset_path, ops_file_path)
    print(f'Loading data from {current_dataset.dataset_path}')

    other_params = {
        'frame_corrector_obj': None,
        # These don't change
        'frame_batch_size': 1000,
        'pixel_batch_size': 5000,
        'num_workers': 0,
        'sim_conf': 5,
        'dtype': 'float32'  # currently only float32 supported
    }
    block_sizes = (pmd_params_dict.pop('block_height'), pmd_params_dict.pop('block_width'))
    # overlap = (pmd_params_dict.pop('overlaps_height'), pmd_params_dict.pop('overlaps_width'))
    frames_to_init = pmd_params_dict.pop('frames_to_init')

    pmd_params_dict.update(other_params)

    # Run PMD
    U, R, s, V, std_img, mean_img, data_shape, data_order = localmd_decomposition(
        current_dataset, block_sizes, frames_to_init, **pmd_params_dict)

    # TODO Attach logger

    # Save the compressed results to a sparse NPZ
    # npz_save_name = ROOT / SESSION_PATH / 'alf' / f'FOV_{FOV:02}' / 'PMD.npz'
    npz_save_name = os.path.join(output_folder, 'PMD.npz')
    print(f'Saving to {npz_save_name}')
    U = U.tocsr()
    np.savez(npz_save_name,
             fov_shape=data_shape[:2], fov_order=data_order, U_data=U.data, U_indices=U.indices,
             U_indptr=U.indptr, U_shape=U.shape, U_format=type(U), R=R, s=s, Vt=V, mean_img=mean_img,
             noise_var_img=std_img)

    # Generate a comparison triptych
    print('Generate a comparison triptych')
    # These two intervals specify what part of the FOV we want to analyze.
    sz = 100  # n pixels in each dim
    center = np.round(np.array(current_dataset.shape[:2]) / 2)
    dim1_interval = [int(center[0] - round(sz / 2)), int(center[0] + round(sz / 2))]
    dim2_interval = [int(center[1] - round(sz / 2)), int(center[1] + round(sz / 2))]

    # Specify which frames you want to see
    start_frame = 0
    end_frame = 2000
    frames = np.arange(start_frame, end_frame)

    output_triptych = generate_PMD_comparison_triptych(
        current_dataset, frames, U, R, s, V, mean_img, std_img, data_order,
        data_shape, dim1_interval, dim2_interval, frame_corrector=None)

    # Save the triptych as a tiff file, which can be viewed in imageJ
    # Modify the filename below as desired
    # parts = SESSION_PATH.split('/')
    # exp_ref = '_'.join([parts[1], str(int(parts[2])), parts[0]])
    # filename_to_save = ROOT / SESSION_PATH / 'alf' / f'FOV_{FOV:02}' / 'mpci.pmdTriptych.tiff'
    filename_to_save = os.path.join(output_folder, 'PMDTriptych.tiff')

    # The below line saves the tiff file
    print(f'Saving triptych to {filename_to_save}')
    tifffile.imwrite(filename_to_save, output_triptych.transpose(2, 0, 1).astype('float32'))




import argparse
import masknmf
import os
import numpy as np
from typing import *
import pathlib
from pathlib import Path

class MotionBinDataset:
    """Load a suite2p data.bin imaging registration file."""

    def __init__(self,
                 data_path: Union[str, pathlib.Path],
                 metadata_path: Union[str, pathlib.Path]):
        """
        Load a suite2p data.bin imaging registration file.

        Parameters
        ----------
        data_path (str, pathlib.Path): The session path containing preprocessed data.
        metadata_path (str, pathlib.Path): The metadata_path to load.
        """
        self.bin_path = Path(data_path)
        self.ops_path = Path(metadata_path)
        self._dtype = np.int16
        self._shape = self._compute_shape()
        self.data = np.memmap(self.bin_path, mode='r', dtype=self.dtype, shape=self.shape)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

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
        return self._shape

    @property
    def ndim(self):
        return len(self.shape)

    def _compute_shape(self):
        """
        Loads the suite2p ops file to retrieve the dimensions of the data.bin file.

        Returns
        -------
        (int, int, int)
            number of frames, number of y pixels, number of x pixels.
        """
        ops_file = self.ops_path
        if ops_file.exists():
            ops = np.load(ops_file, allow_pickle=True).item()
        else:
            raise ValueError("Ops file not found")
        return ops['nframes'], ops['Ly'], ops['Lx']

    def __getitem__(self, item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]]):
        return self.data[item]




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='PMD', description='Run penalized matrix decomposition (PMD) on suite2p file registration bin file.')
    parser.add_argument('--bin_file', type=str, help='The absolute path of the dataset wherever this script is being run')
    parser.add_argument('--ops_file', type=str, help='The absolute path of the ops file wherever this script is being run')
    parser.add_argument('--output_location', type=str, help='The location where the outputs should be saved on the filesystem where this is being run')
    # parser.add_argument('--fov', type=int, help=f'The field of view number')
    parser.add_argument('--batching_height', default=256, type=int,
                        help='The height of each spatial region of data we process in PMD Batch mode')
    parser.add_argument('--batching_width', default=256, type=int,
                        help='The width of each spatial region of data we process in PMD Batch mode')
    parser.add_argument('--batching_overlap', default=10, dtype=int,
                        help='The overlap of adjacent regions when we process data in PMD Batch mode')
    parser.add_argument('--block_height', default=32, type=int,
                        help='The height of the blocks in pixels of the core PMD algorithm (should be comparable to size of somata)')
    parser.add_argument('--block_width', default=32, type=int,
                        help='The width of the blocks in pixels of the core PMD algorithm (should be comparable to size of somata)')
    parser.add_argument('--frames_to_init', default=50000, type=int,
                        help='We begin the method by finding a low-rank (conservative) estimate of the linear subspace in which the signal resides.'
                             'Preferably to use all available frames; set a smaller batch_height and batch_width as needed to avoid memory errors')
    parser.add_argument('--background_rank', default=10, type=int)
    parser.add_argument('--max_consecutive_failures', default=3, type=int)
    parser.add_argument('--max_components', default=20, type=int)
    parser.add_argument('--device', default="cpu", type=str)
    

    pmd_params_dict = vars(parser.parse_args())
    dataset_path = pmd_params_dict.pop('bin_file')
    ops_file_path = pmd_params_dict.pop('ops_file')
    output_folder=pmd_params_dict.pop('output_location')

    # current_dataset = MotionBinDataset(ROOT / SESSION_PATH)
    #This is a memory mapping, so it's very fast
    current_dataset = MotionBinDataset(dataset_path, ops_file_path)

    batching_height, batching_width = (pmd_params_dict.pop('batching_height'), pmd_params_dict.pop('batching_width'))
    batching_overlap = pmd_params_dict.pop('batching_overlap')
    max_components = pmd_params_dict.pop('max_components')
    max_consecutive_failures = pmd_params_dict.pop('max_consecutive_failures')
    background_rank = pmd_params_dict.pop('background_rank')
    device = pmd_params_dict.pop('device')
    block_height, block_width = (pmd_params_dict.pop('block_height'), pmd_params_dict.pop('block_width'))
    frames_to_init = min(current_dataset.shape[0], pmd_params_dict.pop('frames_to_init'))

    pmd_obj = masknmf.compression.pmd_batch(current_dataset,
                                            [batching_height, batching_width],
                                            [batching_overlap, batching_overlap],
                                            [block_height, block_width],
                                            frames_to_init,
                                            max_components=max_components,
                                            max_consecutive_failures=max_consecutive_failures,
                                            background_rank=background_rank,
                                            device=device)
    output_location = os.path.join(output_folder, "pmd_decomposition.npz")
    np.savez(output_location, pmd=pmd_obj)





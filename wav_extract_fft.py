"""Script extracts the frequencies from the dataset and makes frequency "prints" of all .wav files passed as input.

IN: Paths to directories consisting of .wav files.
OUT: Saved .fft.npy files for respective .wav files in input directories.

Run instructions:
python extract-features-FFT.py train_dir_path_1 train_dir_path_2 ... train_dir_path_N

Where train_dir_path_i consists of .wav files.

NOTE:
1. Use ONLY absolute paths. 
"""

import scipy
import scipy.io.wavfile
import os
import sys
import glob
import pathlib
import numpy as np

import warnings
warnings.filterwarnings('ignore')

genres = os.listdir('./converted')
for g in genres:
    pathlib.Path(f'ffc_extracted/{g}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'./converted/{g}'):
        songname = f'./converted/{g}/{filename}'
        sample_rate, song_array = scipy.io.wavfile.read(songname)
        fft_features = abs(scipy.fft.fft(song_array[:10000]))
        #base_fn, ext = os.path.splitext(songname)
        #data_fn = base_fn + ".fft"
        np.save(f'ffc_extracted/{g}/{filename[:-3].replace(".", "")}.fft', fft_features)
    
    



#if __name__ == "__main__":
#	main()
	

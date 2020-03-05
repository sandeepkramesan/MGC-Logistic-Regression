import os
import pathlib
import warnings
warnings.filterwarnings('ignore')


genres = os.listdir('./gtzan')

for g in genres:
    pathlib.Path(f'converted/{g}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(f'./gtzan/{g}'):
        songname = f'./gtzan/{g}/{filename}'
        os.system("ffmpeg -i " + songname + f' ./converted/{g}/{filename[:-3].replace(".", "")}.wav')

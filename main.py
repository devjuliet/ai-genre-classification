import librosa
import os
import numpy as np
import pandas as pd

base_path = './Data/genres_original/'
columns = [f'feat{i}' for i in range(20)]
columns.append('label')
data = pd.DataFrame(columns=columns)

for directory in os.listdir(base_path):
    label = directory
    genre_dir = os.path.join(base_path, directory)
    for file in os.listdir(genre_dir):
        audio_path = os.path.join(base_path, directory, file)
        x, sr = librosa.load(audio_path)
        mfcc = librosa.feature.mfcc(x, sr=sr)
        features = np.mean(mfcc, axis=1)
        features = features.tolist()
        features.append(label)

        df_length = len(data)
        data.loc[df_length] = features

data.to_csv('features.csv')
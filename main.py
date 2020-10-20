import librosa

audio_path = './Data/genres_original/blues/blues.00000.wav'
x, sr = librosa.load(audio_path)
mfcc = librosa.feature.mfcc(x, sr=sr)

print(mfcc.shape)
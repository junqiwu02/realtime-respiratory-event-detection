import numpy as np
from imblearn.over_sampling import RandomOverSampler

import glob
import pickle
from tqdm.notebook import tqdm

import librosa

RATE = 16000
FEAT_DIM = 40

class Preprocess:

    def __init__(self):
        test_neg = {f: 0 for f in glob.glob('data/test/not_sick/*.wav')}
        test_pos = {f: 1 for f in glob.glob('data/test/sick/*.wav')}
        self.test = {**test_neg, **test_pos}

        train_neg = {f: 0 for f in glob.glob('data/train/not_sick/*.wav')}
        train_pos = {f: 1 for f in glob.glob('data/train/sick/*.wav')}
        self.train = {**train_neg, **train_pos}

        val_neg = {f: 0 for f in glob.glob('data/validation/not_sick/*.wav')}
        val_pos = {f: 1 for f in glob.glob('data/validation/sick/*.wav')}
        self.val = {**val_neg, **val_pos}

    def get_test(self):
        return self.get(self.test)

    def get_train(self):
        return self.get(self.train)

    def get_val(self):
        return self.get(self.val)

    def get(self, data):
        feats = []
        labels = []
        for f, l in tqdm(data.items()):
            y, sr = librosa.load(f, sr=RATE, res_type='kaiser_fast')
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=FEAT_DIM)
            mfccs = np.mean(mfccs.T, axis=0) # take the mean of each mfcc across the time series

            feats.append(mfccs)
            labels.append(l)
        
        X = np.array(feats)
        y = np.array(labels)

        return X, y

def resample(X, y):
    X, y = RandomOverSampler().fit_resample(X, y)
    
    return X, y

def pickle_dump(data, path):
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def pickle_load(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data
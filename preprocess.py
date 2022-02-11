import numpy as np
from imblearn.over_sampling import RandomOverSampler

import glob
import pickle
from tqdm.notebook import tqdm

import librosa

RATE = 16000
FEAT_DIM = 40
TENSOR_LEN = 156

class Preprocess:

    def __init__(self):
        self.test_neg_files = glob.glob('data/test/not_sick/*.wav')
        self.test_pos_files = glob.glob('data/test/sick/*.wav')

        self.train_neg_files = glob.glob('data/train/not_sick/*.wav')
        self.train_pos_files = glob.glob('data/train/sick/*.wav')

        self.val_neg_files = glob.glob('data/validation/not_sick/*.wav')
        self.val_pos_files = glob.glob('data/validation/sick/*.wav')

    def get_test(self, max_inputs=None):
        return self.get(self.test_neg_files, self.test_pos_files, max_inputs)

    def get_train(self, max_inputs=None):
        return self.get(self.train_neg_files, self.train_pos_files, max_inputs)

    def get_val(self, max_inputs=None):
        return self.get(self.val_neg_files, self.val_pos_files, max_inputs)

    def get(self, neg_files, pos_files, max_inputs=None):
        seqs = []
        labels = []
        for f in tqdm(neg_files[:max_inputs]):
            y, sr = librosa.load(f, sr=RATE, res_type='kaiser_fast')
            feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=FEAT_DIM)
            feat = np.mean(feat.T, axis=0)

            seqs.append(feat)
            labels.append(0)

        for f in tqdm(pos_files[:max_inputs]):
            y, sr = librosa.load(f, sr=RATE, res_type='kaiser_fast')
            feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=FEAT_DIM)
            feat = np.mean(feat.T, axis=0)

            seqs.append(feat)
            labels.append(1)

        # resize all sequences to the avg length
        # tensor_len = sum(map(len, seqs)) // len(seqs)
        # tensor_len = TENSOR_LEN

        # resized = []
        # for x in seqs:
        #     res = None
        #     if len(x) < tensor_len:
        #         res = np.zeros((tensor_len, FEAT_DIM))
        #         res[:x.shape[0], :x.shape[1]] = x
        #     else:
        #         res = np.array(x[:tensor_len])
        #     resized.append(res)
        # X = np.array(resized)

        # print(f'Resized to {X[0].shape}!')

        # normalize
        # X = X / np.linalg.norm(X)
        
        X = np.array(seqs)
        y = np.array(labels)

        return X, y

def resample(X, y):
    # flatten
    X = X.reshape(len(X), -1)

    X, y = RandomOverSampler().fit_resample(X, y)

    X = X.reshape(len(X), -1, FEAT_DIM)

    return X, y

def one_hot(y):
    Y = np.zeros((len(y), 2))
    Y[np.arange(len(y)), y] = 1

    return Y

def pickle_dump(data, path):
    with open(path, 'wb') as file:
        pickle.dump(data, file)

def pickle_load(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data
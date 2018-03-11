import json
import numpy as np

from keras.utils import to_categorical


def load_data(one_hot=False):
    with open('data/shipsnet.json',) as f:
        d = json.load(f)
        X, y = np.array(d['data'], dtype='float'), np.array(d['labels'])

    X /= 255
    X = X.reshape(-1, 3, 80, 80).transpose([0, 2, 3, 1])

    if one_hot:
        y = to_categorical(y)

    return X, y
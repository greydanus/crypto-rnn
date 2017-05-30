# Decoding the Enigma Cipher with Recurrent Neural Networks
# Sam Greydanus. January 2017. MIT License.

import copy
from crypto_enigma import *
import numpy as np

# enigma data class
class Dataloader():
    def __init__(self, A, tsteps, key_len):
        self.A = A
        self.tsteps = tsteps
        self.key_len = key_len
    
    def encode(self, key, text):
        key = key.decode('unicode-escape')
        enigma = EnigmaConfig.config_enigma(rotor_names=u"A-I-II-III", window_letters=key, \
                                 plugs=u"", rings=u"02.14.08")
        return enigma.enigma_encoding(text)
    
    def rands(self, size):
        ix = np.random.randint(len(self.A),size=size)
        return ''.join([self.A[i] for i in ix])
    
    def one_hot(self, s):
        ix = [self.A.find(l) for l in s]
        z = np.zeros((len(s),len(self.A)))
        z[range(len(s)),ix] = 1
        return z
    
    def next_batch(self, batch_size, info=False):
        batch_X = [] ; batch_y = [] ; batch_Xs = [] ; batch_ks = [] ; batch_ys = []
        for _ in range(batch_size):
            ys = self.rands(self.tsteps).decode('unicode-escape')
            ks = self.rands(self.key_len)
            Xs = self.encode(ks, ys)
            X = self.one_hot(ks + Xs)
            y = self.one_hot(ks + ys)
            batch_X.append(X)
            batch_y.append(y)
            batch_Xs.append(Xs) ; batch_ys.append(ys) ; batch_ks.append(ks)
        
        if not info:
            return (batch_X,batch_y)
        else:
            return (batch_X, batch_y, batch_Xs, batch_ks, batch_ys)

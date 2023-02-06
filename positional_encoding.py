#Postional Ecoding provisdes information about the positoin of a word or token in a sequence to the model. Gives a
#mathematical represeantation of the position of token in sequence
# in transformers, postional ecoding vector is added to the token emobedding vextor to provide
# model with infromation about the position of word in sequence. Positional encoding vector
#are typically sin function of the position index, with different frequency for differenct 
#dimension of the vector. This gives the elative position of token. 
# Way of encosing the posioton of a word. 


import pickle
import matplotlib.pyplot as plt
import numpy as np

def positional_ecoding_matrix(L, d, n = 10000):
    assert d % 2 == 0
    d2 = d//2 

    P = np.zeros((L,d))
    k = np.arange(L).reshape(-1,1)
    i = np.arange(d2).reshape(1,-1)
    denom = np.power(n, -i/d2)
    args = k* denom
    P[:, ::2] = np.sin(args)
    P[:, 1::2] = np.cos(args)
    return P

pos_matrix = positional_ecoding_matrix(L=2048, d = 512)
assert pos_matrix.shape == (2048, 512)
plt.pcolormesh(pos_matrix, cmap= 'RdBu')
plt.xlabel('Depth')
plt.ylabel('Postion')
plt.colorbar()
plt.show() 

with open("posenc-2048-512.pickle", "wb") as fp:
    pickle.dump(pos_matrix, fp)
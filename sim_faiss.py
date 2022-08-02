import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

import faiss                   # make faiss available

# Flat
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained) # True
index.add(xb)                  # add vectors to the index
print(index.ntotal) # 100000

k = 4                          # we want to see 4 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print(I)
'''
[[  0 393 363  78]
 [  1 555 277 364]
 [  2 304 101  13]
 [  3 173  18 182]
 [  4 288 370 531]]
'''
print(D)
'''
[[0.        7.175174  7.2076287 7.251163 ]
 [0.        6.323565  6.684582  6.799944 ]
 [0.        5.7964087 6.3917365 7.2815127]
 [0.        7.277905  7.5279875 7.6628447]
 [0.        6.763804  7.295122  7.368814 ]]
'''
D, I = index.search(xq, k)     # actual search
print(I[:5])                   # neighbors of the 5 first queries
'''
[[ 381  207  210  477]
 [ 526  911  142   72]
 [ 838  527 1290  425]
 [ 196  184  164  359]
 [ 526  377  120  425]]
'''
print(I[-5:])                  # neighbors of the 5 last queries
'''
[[ 9900 10500  9309  9831]
 [11055 10895 10812 11321]
 [11353 11103 10164  9787]
 [10571 10664 10632  9638]
 [ 9628  9554 10036  9582]]
'''

# IVFFlat
nlist = 100
quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

assert not index.is_trained
index.train(xb)
assert index.is_trained

index.add(xb)                  # add may be a bit slower as well
D, I = index.search(xq, k)     # actual search
print(I[-5:])                  # neighbors of the 5 last queries
'''
[[ 9900  9309  9810 10048]
 [11055 10895 10812 11321]
 [11353 10164  9787 10719]
 [10571 10664 10632 10203]
 [ 9628  9554  9582 10304]]
'''
index.nprobe = 10              # default nprobe is 1, try a few more
D, I = index.search(xq, k)
print(I[-5:])                  # neighbors of the 5 last queries
'''
[[ 9900 10500  9309  9831]
 [11055 10895 10812 11321]
 [11353 11103 10164  9787]
 [10571 10664 10632  9638]
 [ 9628  9554 10036  9582]]
'''

# IVFOQ
nlist = 100
m = 8
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                  # 8 specifies that each sub-vector is encoded as 8 bits
index.train(xb)
index.add(xb)
D, I = index.search(xb[:5], k) # sanity check
print(I)
'''
[[   0   78  424  159]
 [   1  555  706 1063]
 [   2  179  304  134]
 [   3   64  773    8]
 [   4  288  827  531]]
'''
print(D)
'''
[[1.5882268 6.331396  6.440189  6.473257 ]
 [1.274326  5.728371  6.056792  6.1539173]
 [1.7501019 6.1581926 6.310023  6.365546 ]
 [1.8521194 6.6665597 6.978093  6.9924507]
 [1.5939493 5.717939  6.3486733 6.374599 ]]
'''
index.nprobe = 10              # make comparable with experiment above
D, I = index.search(xq, k)     # search
print(I[-5:])
'''
[[ 8746  9966  9853  9968]
 [11373 10913 10240 10403]
 [11291 10719 10494 10424]
 [10122 10005 11276 11578]
 [ 9644  9905 10370  9229]]
'''
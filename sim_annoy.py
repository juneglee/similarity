# Annoy (Approximate Nearest Neighbor Oh Yeah)

from annoy import AnnoyIndex
import random

f = 40  # Length of item vector that will be indexed
'''
AnnoyIndex(f, metric) returns a new index that's read-write and stores vector of f dimensions.
Metric can be "angular", "euclidean", "manhattan", "hamming", or "dot
angular : cosine distance
'''
t = AnnoyIndex(f, 'angular')
for i in range(1000):
    v = [random.gauss(0, 1) for z in range(f)]
    '''
    a.add_item(i, v) adds item i (any nonnegative integer) with vector v.
    Note that it will allocate memory for max(i)+1 items
    '''
    t.add_item(i, v)

'''
a.build(n_trees, n_jobs=-1) builds a forest of n_trees trees.
More trees gives higher precision when querying.
After calling build, no more items can be added.
n_jobs specifies the number of threads used to build the trees. n_jobs=-1 uses all available CPU cores
'''
t.build(10, n_jobs= -1) # 10 trees
# a.save(fn, prefault=False) saves the index to disk and loads it (see next function).
# After saving, no more items can be added.
t.save('weight/test.ann')

# ...loading

u = AnnoyIndex(f, 'angular')
'''
a.load(fn, prefault=False) loads (mmaps) an index from disk.
If prefault is set to True, it will pre-read the entire file into memory (using mmap with MAP_POPULATE).
Default is False
'''
u.load('weight/test.ann') # super fast, will just mmap the file
'''
a.get_nns_by_item(i, n, search_k=-1, include_distances=False) returns the n closest items.
During the query it will inspect up to search_k nodes which defaults to n_trees * n if not provided.
search_k gives you a run-time tradeoff between better accuracy and speed.
If you set include_distances to True,
it will return a 2 element tuple with two lists in it: the second one containing all corresponding distances
'''
print(u.get_nns_by_item(0, 1000)) # will find the 1000 nearest neighbors
print(u.get_n_trees())

# other
'''
- a.get_nns_by_vector(v, n, search_k=-1, include_distances=False) same but query by vector v.
- a.get_item_vector(i) returns the vector for item i that was previously added.
- a.get_distance(i, j) returns the distance between items i and j. NOTE: this used to return the squared distance, 
    but has been changed as of Aug 2016.
- a.get_n_items() returns the number of items in the index.
- a.get_n_trees() returns the number of trees in the index.
- a.on_disk_build(fn) prepares annoy to build the index in the specified file instead of RAM 
    (execute before adding items, no need to save after build)
- a.set_seed(seed) will initialize the random number generator with the given seed. 
    Only used for building up the tree, i. e. only necessary to pass this before adding the items. 
    Will have no effect after calling a.build(n_trees) or a.load(fn).
'''


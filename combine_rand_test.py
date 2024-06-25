# cc -fPIC -shared -o libms.so midsq.c

import ctypes
import numpy as np

np.random.seed(0)

n = 1_000
save = True

def combine_rands(a, b):
    maxval = np.iinfo(np.uint32).max + 2 # +2 to ensure values can never be identically zero
    a1 = (a*1e9).astype(np.uint32)
    b1 = (b*1e9).astype(np.uint32)
    c = np.bitwise_xor(a1*b1, a1-b1)
    c = (1+c.astype(float))/maxval
    return c

def combine_rands_djk(a, b):
    c = np.bitwise_xor(a*b, a-b).astype(np.uint64)
    d = c / np.iinfo(np.uint64).max
    return d, c

def just_xor(a, b):
    c = np.bitwise_xor(a, b)
    d = c / np.iinfo(np.uint64).max
    return d, c

def diff(a, b):
    c = b-a
    d = c / np.iinfo(np.uint64).max
    return d, c

x = np.random.rand(1)
y = np.random.rand(1)
z = combine_rands(x,y)


x = np.random.randint(low=np.iinfo('int64').min, high=np.iinfo('int64').max, dtype=np.int64, size=n)
y = np.random.randint(low=np.iinfo('int64').min, high=np.iinfo('int64').max, dtype=np.int64, size=n)
z, ints = combine_rands_djk(x,y)

# Ctypes
x = np.random.randint(size=n, low=np.iinfo(np.uint64).max, dtype=np.uint64)
y = np.random.randint(size=n, low=np.iinfo(np.uint64).max, dtype=np.uint64)

# load the library
mylib = ctypes.CDLL("libms.so")

# C-type corresponding to numpy array 
#ND_POINTER = np.ctypeslib.ndpointer(dtype=np.uint64, ndim=1, flags="C")

# define prototypes
#mylib.mid_sq.argtypes = [ND_POINTER, ND_POINTER, ctypes.c_size_t]

mylib.mid_sq.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),  ctypes.c_size_t]
mylib.mid_sq.restype = ctypes.POINTER(ctypes.c_uint32 * len(x))

#mylib.mid_sq.restype = ND_POINTER

# call function
xi = x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
yi = y.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
zo = mylib.mid_sq(xi, yi, x.size)
z = np.frombuffer(zo.contents, dtype=np.uint32) / np.iinfo(np.uint32).max

mylib.freeArray.argtypes = [ctypes.POINTER(ctypes.c_uint32 * len(z))]
# free buffer
mylib.freeArray(zo)


#z, zints = just_xor(x,y)
#z, zints = diff(x,y)
import matplotlib.pyplot as plt
plt.hist(z, bins=1000)

#if save:
#    strs = ''.join([np.binary_repr(i) for i in zints])
#    sc.savetext(filename='zstrs.txt', string=strs)

plt.show()
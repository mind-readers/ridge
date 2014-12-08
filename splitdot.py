import numpy as np

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import scikits.cuda.linalg as linalg
import scikits.cuda.misc as misc
linalg.init()


GPU_MAX_MEM = ((12884705280*3)/4)

def splitdot_gpu(m1, m2, split=2):
    # Make sure the split factor is sane
    if split > m1.shape[0] or split > m2.shape[1]:
        raise ValueError("Matrices not big enough for split factor")

    # split the array
    m1_split = np.array_split(m1, split, axis=0)
    m2_split = np.array_split(m2, split, axis=1)
    forward = True
    # Iterate over and do partial dots
    final_product = None
    m1_part_gpu = None
    m2_part_gpu = None
    for m1_part in m1_split:
        hchunk = None
        for m2_part in m2_split:
            # If first matmul, swap m1_part and assign hchunk directly
            if hchunk == None:
                # If this is the VERY first iteration, need to ship
                # m1_part AND m2_part
                if m2_part_gpu is None:
                    m2_part_gpu = gpuarray.to_gpu(np.copy(m2_part), order='C')
                # GPU: SWAP OUT m1_part 
                # First delete the old one if it's allocated on GPU
                if m1_part_gpu is not None:
                    del m1_part_gpu
                # Then ship the new one
                m1_part_gpu = gpuarray.to_gpu(np.copy(m1_part), order='C')
                # Then do the computation
                hchunk_gpu = linalg.dot(m1_part_gpu, m2_part_gpu)
                # Fetch the chunk, directly assign because first in row
                hchunk = hchunk_gpu.get()
                del hchunk_gpu
                #hchunk = np.dot(m1_part, m2_part)
            # Else swap m2_part and append to hchunk
            else:
                # GPU: SWAP OUT m2_part
                # GPU: Do matmul of parts and fetch
                # First dealloc the old m2_part
                del m2_part_gpu
                # Then ship the new one
                m2_part_gpu = gpuarray.to_gpu(np.copy(m2_part), order='C')
                # Then do the computation
                fragment_gpu = linalg.dot(m1_part_gpu, m2_part_gpu)
                # Then fetch the fragment
                fragment = fragment_gpu.get()
                # Dealloc the fragment on GPU
                del fragment_gpu
                # We are traversing forward through the row
                if forward:
                    hchunk = np.hstack((hchunk, fragment))
                # We are traversing backward throught the row
                else:
                    hchunk = np.hstack((fragment, hchunk))
        # If first row, directly assign
        if final_product == None:
            final_product = hchunk
        # Else append to bottom
        else:
            final_product = np.vstack((final_product, hchunk))
        # Reverse the way we iterate through the right chunks
        m2_split.reverse()
        # Inicate a reversal of traversal direction
        forward = (not forward)
    del m1_part_gpu
    del m2_part_gpu
    return final_product

# Assume both are column major
def left_dot_col_major_gpu(m1, m2):
    out_gpu = gpuarray.GPUArray((m1.shape[0], m2.shape[1]), np.float64, allocator=cuda.mem_alloc, order='F')
    gpu_mem_avail = GPU_MAX_MEM - out_gpu.nbytes - m1.nbytes
    print("GPU MEM: " + str(GPU_MAX_MEM))
    print("GPU MEM AVAIL: " + str(gpu_mem_avail))
    print("M1 SIZE: " + str(m1.nbytes))
    print("OUT SIZE: " + str(out_gpu.nbytes))
    frags = 1
    while ((m2.nbytes / frags) > gpu_mem_avail):
        frags += 1
    m1_gpu = gpuarray.to_gpu(m1)
    subm2_gpu = None
    shift = 0
    for subm2 in np.array_split(m2, frags, axis=1):
        if subm2_gpu is not None:
            del subm2_gpu
        subm2_gpu = gpuarray.to_gpu(subm2)
        linalg.dot(m1_gpu, subm2_gpu, out=out_gpu[:,shift:shift+subm2.shape[1]])
        shift += subm2.shape[1]
    out = out_gpu.get()
    del m1_gpu
    del out_gpu
    return out

def split_balancer(shape1, shape2):
    gpumaxmem=12884705280/2
    matrixsize = 2
    finalsplit = 1
    while (True):
        size1 = (shape1[0]/finalsplit) * shape1[1] * 8
        size2 = shape2[0] * (shape2[1]/finalsplit) * 8
        size3 = (shape1[0]/finalsplit) * (shape2[1]/finalsplit) * 8
        if gpumaxmem > (size1 + size2 +size3):
            return finalsplit
        finalsplit +=1
    return finalsplit

def dot_on_gpu(m1, m2):
    split = split_balancer(m1.shape, m2.shape)
    return splitdot_gpu(m1, m2, split)



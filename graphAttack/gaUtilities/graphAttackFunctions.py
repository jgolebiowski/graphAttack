import numpy
matmul = numpy.matmul

# # ------ If you wish to unleash your GPU
# # ------ Comment out the lines above ^
# # ------ Uncomment lines below 
# # ------ Needs pyCUDA as well as scikit-cuda

# import pycuda.autoinit
# import pycuda.gpuarray as gpuarray
# import numpy as np
# import skcuda.linalg
# skcuda.linalg.init()

# def matmul(a, b):
#     a = a.astype(np.float32, order="C")
#     b = b.astype(np.float32, order="C")
#     a_gpu = gpuarray.to_gpu(a)
#     b_gpu = gpuarray.to_gpu(b)

#     c_gpu = skcuda.linalg.dot(a_gpu, b_gpu)
#     return c_gpu.get()
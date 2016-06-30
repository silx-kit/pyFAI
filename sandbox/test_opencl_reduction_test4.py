import pyopencl as cl
from pyopencl import array
import numpy

length = 640000
workgroup_size = 128

a = numpy.random.rand(length).astype(numpy.float32)
a.shape = (length/8,4,2)
input_a = a.reshape(length)

min0 = a[:, :, 0].min()
max0 = a[:, :, 0].max()
min1 = a[:, :, 1].min()
max1 = a[:, :, 1].max()
minmax=(min0,max0,min1,max1)


platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
ctx = cl.Context((device,))
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags


d_input     = array.to_device(queue, input_a)
d_preresult = cl.Buffer(ctx, mf.READ_WRITE, 4*4*workgroup_size)
d_result = cl.Buffer(ctx, mf.READ_WRITE, 4*4)

with open("pyFAI/resources/openCL/reduction_test4.cl", "r") as kernelFile:
    kernel_src = kernelFile.read()
    kernel_src = kernel_src.replace("#include \"for_eclipse.h\"", "")

compile_options = "-D WORKGROUP_SIZE=%i" % (workgroup_size)

program = cl.Program(ctx, kernel_src).build(options=compile_options)

program.reduce1(queue, (workgroup_size*workgroup_size,), (workgroup_size,), d_input.data,  numpy.uint32(length), d_preresult)

program.reduce2(queue, (workgroup_size,), (workgroup_size,), d_preresult, d_result)

result = numpy.ndarray(4,dtype=numpy.float32)

cl.enqueue_copy(queue,result, d_result)


print minmax

print result





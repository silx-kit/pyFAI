:Author: Jérôme Kieffer
:Date: 21/10/2021
:Keywords: Tutorials
:Target: Advanced users tutorials using jupyter notebooks

.. _paallelization:

Parallelization
===============

There are at least 6 different layouts for integrating in
parallel large stacks of images which can be ordered from the
simplest to the most technical:

0. Naive serial loop with parallel decompression (thanks to OpenMP in
bitshuffle-LZ4) and parallel integration (thanks to OpenMP, CSR matrix
multiplication). This approach was developed at a time computers had
few cores (~4) but the way OpenMP is used here does not scale at all
and it is often slower than the serial version on "many-core" systems
(modern servers featuring more than 32 cores).

1. Multithreading: While threads are known to be inefficient in Python
due to the GIL, this pattern can be effective when all code is GIL-free.
One could design a pipeline with one reader using direct-chunk-read and a
pool of threads performing decompression + integration. Decompression
must be tuned to use a single thread to avoid cache poisoning.
One of the limitation could be the repetitive allocation of buffers for
output, but maybe python is able to handle this transparently.

2. Multiprocessing: This is known to be efficient under Linux thanks to
the `fork` mechanism but one has to use `spawn`, like under Windows to
be compatible with GPU processing. One GPU can host up to 15 parallel instances.

3. Dask/Joblib: this is a variant of the former, maybe a bit simpler to
implement, but still a lot of things to tune.

4. Full GPU processing: this requires the hardware (starts ~10k€) and
some software blocks like the LZ4 decompression, the bitshuffling and
the azimuthal integration to be all performed on the GPU. The advantage
is that one benefits from the direct-chunk-read from HDF5, transfers
little data to the GPU. Probably one of the best solution but it
requires highly skilled staff to make it run.

5. Full FPGA processing: It is proven to be up to 10x faster compared
to GPU but requires even more specialized hardware and staff.

The notebook hereafter present some of those approaches:

.. toctree::
   :maxdepth: 1

   Direct_chunk_read

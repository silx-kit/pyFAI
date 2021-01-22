:Author: Jérôme Kieffer
:Date: 22/01/2021
:Keywords: Performance analysis
:Target: user

.. _pyfaibenchmark:

pyFAI-benchmark
===============

Measure the speed for azimuthal integration.

Purpose
-------

Measures the avarage execution time for azimuthal integration with various image sizes and algorithms,
can be used to select the most suitable integrator, to evaluate the perfomance of a given computer, debug 
some hardware problems.

Image are between 1 and 16 Mpixel in size and all the method tested are providing the same
results for azimuthal integration (and validated). 
The `bbox` pixel splitting schem is used which is also the default one.
By default, only the `histogram` and `CSR` algorithm (implemented in cython) are measued, but 
OpenCL devices can be probed with options "-c", "-g" and "-a".

The result is a graphic with the number of images integrated per second as function 
of the image size. 
All the corresponding timings are also recorded in a JSON file.
 
Since pyFAI version 0.20, whith the new generation of integrator put in production,
both ``integrate1d_legacy`` and ``integrate1d_ng`` are benchmarked together to validate 
the absence of performance regression.
A factor larger than 2 sould be considered as a bug.  

Usage 
-----

pyFAI-bencmark [options]

optional arguments:
  -h, --help            show the help message and exit
  -v, --version         show program's version number and exit
  -d, --debug           switch to verbose/debug mode
  -c, --cpu             perform benchmark using OpenCL on the CPU
  -g, --gpu             perform benchmark using OpenCL on the GPU
  -a, --acc             perform benchmark using OpenCL on the Accelerator (like XeonPhi/MIC)
  -s SIZE, --size SIZE  Limit the size of the dataset to X Mpixel images (for computer with limited memory)
  -n NUMBER, --number NUMBER
                        Number of repetition of the test (or time used for each test), by default 10 (sec)
  -2d, --2dimention     Benchmark also algorithm for 2D-regrouping
  --no-1dimention       Do not benchmark algorithms for 1D-regrouping
  -m, --memprof         Perfrom memory profiling (Linux only)
  -r REPEAT, --repeat REPEAT
                        Repeat each benchmark x times to take the best
Result
------

The typical result looks like this:

.. figure:: ../img/benchmark.svg
   :align: center
   :alt: benchmark performed on a 2016 single-socket workstation and a gaming graphics card.

For the detailed explaination on how to read this plot, please refer to :ref:`performances`.

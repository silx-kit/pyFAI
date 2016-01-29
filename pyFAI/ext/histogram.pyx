IF HAVE_OPENMP:
    include "histogram_omp.pxi"
ELSE:
    include "histogram_nomp.pxi"

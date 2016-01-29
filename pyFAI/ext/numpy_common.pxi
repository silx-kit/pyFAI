# hack to avoid C compiler warnings about unused functions in the NumPy header files
# Sources: Cython test suite.

cdef extern from *:
    bint FALSE "0"
    void import_array()
    void import_umath()

if FALSE:
    import_array()
    import_umath()

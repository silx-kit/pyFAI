#!/usr/bin/python
import numpy
a=numpy.arange(100)-50.
g=numpy.exp(-a*a/5000)
gg=numpy.outer(g,g)
from pyFAI import bilinear
b=bilinear.Bilinear(gg)

ok = 0
for s in range(1000):
    i,j=numpy.random.randint(100),numpy.random.randint(100)
    k,l=b.local_maxi((i,j),1)
    if (k,l)==(i,j): 
        print "same",i,j
    elif abs(k - 50) > 1e-4 or abs(l - 50) > 1e-4:
        print "error", i, j, k, l
    else: 
        print "OK",i,j,k,l
        ok+=1
print "result: %.1f"%(ok/10.)

:Author: Jérôme Kieffer
:Date: 24/06/2022
:Keywords: average sigma uncertainties standard-deviation standard-error-of-the-mean std sem
:Target: General audiance

Weighted average and uncertainties propagation in pyFAI 
=======================================================

TODO

Distinction between the *std* and the *sem*  

A word about notation
---------------------

Five different histograms are calculated for an integration:
sum_signal
sum_variance
sum_normalisation
sum_normalisation_sqarred
sum_count 


Average
-------

sum_signal/sum_normalisation

Uncertainties propagated from known variance
--------------------------------------------
Often assuming Poisson statistics


Uncertainties calculated from the variance in a ring
----------------------------------------------------

Note the difference with schubert's paper where wi are not squarred

Conclusion
----------
Point to a notebook where the demonstration is performed that average/uncertainties scale well with normalization_factor 

Reference:
----------
[1] https://ui.adsabs.harvard.edu/#abs/1969drea.book.....B/abstract
[2] https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
[3] https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf

Preprocessing tool: pyFAI-average
=================================

Purpose
-------

This tool is used to average out a set of dark current images using
mean or median filter (along the image stack). One can also reject outliers
be specifying a cutoff (remove cosmic rays / zingers from dark)

It can also be used to merge many images from the same sample when using a small beam
and reduce the spotty-ness of Debye-Scherrer rings. In this case the "max-filter" is usually
recommended.

Options:
--------

Usage: pyFAI-average [options] -o output.edf file1.edf file2.edf ...

positional arguments:
  FILE                  Files to be processed

optional arguments:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
  -o OUTPUT, --output OUTPUT
                        Output/ destination of average image
  -m METHOD, --method METHOD
                        Method used for averaging, can be 'mean'(default) or
                        'min', 'max', 'median', 'sum', 'quantiles'
  -c CUTOFF, --cutoff CUTOFF
                        Take the mean of the average +/- cutoff * std_dev.
  -F FORMAT, --format FORMAT
                        Output file/image format (by default EDF)
  -d DARK, --dark DARK  Dark noise to be subtracted
  -f FLAT, --flat FLAT  Flat field correction
  -v, --verbose         switch to verbose/debug mode
  -q QUANTILES, --quantiles QUANTILES
                        average out between two quantiles -q 0.20-0.90


.. command-output:: pyFAI-average --help
    :nostderr:


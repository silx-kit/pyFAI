Preprocessing tool: pyFAI-average
=================================

Purpose
-------

This tool can be used to average out a set of dark current images using
mean or median filter (along the image stack). One can also reject outliers
be specifying a cutoff (remove cosmic rays / zingers from dark)

It can also be used to merge many images from the same sample when using a small beam
and reduce the spotty-ness of Debye-Sherrer rings. In this case the "max-filter" is usually
recommended.

Options:
--------

Usage: pyFAI-average [options] -o option.edf file1.edf file2.edf ...

Options:
  --version             show program's version number and exit
  -h, --help            show help message and exit
  -o OUTPUT, --output=OUTPUT
                        Output/ destination of average image
  -m METHOD, --method=METHOD
                        Method used for averaging, can be 'mean'(default) or
                        'median', 'min' or 'max'
  -c CUTOFF, --cutoff=CUTOFF
                        Take the mean of the average +/- cutoff * std_dev.
  -f FORMAT, --format=FORMAT
                        Output file/image format (by default EDF)

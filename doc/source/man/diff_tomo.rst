Integration tool: diff_tomo
===========================

Purpose
-------

Azimuthal integration for diffraction tomography.

Diffraction tomography is an experiment where 2D diffraction patterns are recorded 
while performing a 2D scan, one (the slowest) in rotation around the sample center
and the other (the fastest) along a translation through the sample.
Diff_tomo is a script (based on pyFAI and h5py) which allows the reduction of this 
4D dataset into a 3D dataset containing the rotations angle (hundreds), the translation step (hundreds)
and the many diffraction angles (thousands). The resulting dataset can be opened using PyMca roitool
where the 1d dataset has to be selected as last dimension. This file is not (yet) NeXus compliant.

This tool can be used for mapping experiments if one considers the slow scan direction as the rotation.

tips: If the number of files is too large, use double quotes around "*.edf" 


Usage:
------

diff_tomo [options] -p ponifile imagefiles*

Options:
--------

  --version             show program's version number and exit
  -h, --help            show help message and exit
  -o FILE, --output=FILE
                        HDF5 File where processed sinogram was saved
  -v, --verbose         switch to verbose/debug mode
  -P FILE, --prefix=FILE
                        Prefix or common base for all files
  -e EXTENSION, --extension=EXTENSION
                        Process all files with this extension
  -t NTRANS, --nTrans=NTRANS
                        number of points in translation
  -r NROT, --nRot=NROT  number of points in rotation
  -c NDIFF, --nDiff=NDIFF
                        number of points in diffraction powder pattern
  -d FILE, --dark=FILE  list of dark images to average and subtract
  -f FILE, --flat=FILE  list of flat images to average and divide
  -m FILE, --mask=FILE  file containing the mask
  -p FILE, --poni=FILE  file containing the diffraction parameter (poni-file)
  -O OFFSET, --offset=OFFSET
                        do not process the first files
  -g, --gpu             process using OpenCL on GPU

Most of those options are mandatory to define the structure of the dataset.

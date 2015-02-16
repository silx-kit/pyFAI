#!/usr/bin/python
# coding: UTF-8
# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import os
import sys
import h5py
import numpy
import fabio

def extract_mask(infile):
    """
    Retrieve the mask from an Eiger master file.

    @param infile: name of the Eiger master file
    """
    h = h5py.File(infile, "r")
    entry = h["entry"]
    instrument = entry["instrument"]
    detector = instrument["detector"]
    detectorSpecific = detector["detectorSpecific"]
    return detectorSpecific["pixel_mask"].value

if __name__ == "__main__":
    infile = os.path.abspath(sys.argv[1])
    if len(sys.argv)>2:
        outfile = sys.argv[2]
    else:
        outfile = os.path.splitext(infile)[0]+"_mask.edf"
    mask = extract_mask(infile)
    if outfile.endswith("msk"):
        fabio.fit2dmaskimage.fit2dmaskimage(data=mask).write(outfile)
    elif outfile.endswith("tif"):
        fabio.tifimage.tifimage(data=mask).write(outfile)
    else:
        fabio.edfimage.edfimage(header={"data_file":infile},data=mask).write(outfile)


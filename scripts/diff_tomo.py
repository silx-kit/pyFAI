#!/usr/bin/python
import logging, time, sys, os, posixpath
import numpy
import fabio
import h5py
import pyFAI
logger = logging.getLogger("diff_tomo")



class DiffTomo(object):
    """
    Basic class for diffraction tomography using pyFAI
    """
    def __init__(self, nTrans=1, nRot=1, nDiff=1000):
        """
        Contructor of the class

        @param nTrans: number of translations
        @param nRot: number of translations
        @param nDiff: number of points in diffraction pattern
        """
        self.nTrans = nTrans
        self.nRot = nRot
        self.nDiff = nDiff
        self.offset = 0
        self.poni = None
        self.ai = None
        self.dark = None
        self.flat = None
        self.mask = None
        self.I0 = None
        self.hdf5 = None
        self.hdf5path = "DiffTomo/NXdata/sinogram"
        self.group = None
        self.dataset = None
        self.inputfiles = []
        self.timing = []
        self.use_gpu = False

    def __repr__(self):
        return "Diffraction Tomography with r=%s t: %s, d:%s" % (self.nRot, self.nTrans, self.nDiff)

    def parse(self):
        """
        parse options from command line
        """
        from optparse import OptionParser
        parser = OptionParser()
        parser.add_option("-o", "--out", dest="outfile",
                          help="HDF5 File where processed sinogram is is saved", metavar="FILE", default="diff_tomo.h5")
        parser.add_option("-v", "--verbose",
                          action="store_true", dest="verbose", default=False,
                          help="switch to verbose mode")
        parser.add_option("-e", "--extension", dest="extension",
                      help="process all files with this extension", default="edf")
        parser.add_option("-t", "--nTrans", dest="nTrans",
                      help="number of points in translation", default=None)
        parser.add_option("-r", "--nRot", dest="nRot",
                      help="number of points in rotation", default=None)
        parser.add_option("-c", "--nDiff", dest="nDiff",
                      help="number of points in diffraction", default=None)
        parser.add_option("-d", "--dark", dest="dark",
                      help="list of dark images to average and subtract", default=None)
        parser.add_option("-f", "--flat", dest="flat",
                      help="list of flat images to average and divide", default=None)
        parser.add_option("-m", "--mask", dest="mask",
                      help="file containing the mask", default=None)
        parser.add_option("-p", "--poni", dest="poni", metavar="FILE",
                      help="file containing the diffraction parameter (poni-file)", default=None)
        parser.add_option("-O", "--offset", dest="offset",
                      help="do not process the first files", default=None)
        parser.add_option("-g", "--gpu", dest="gpu", action="store_true",
                    help="process using OpenCL on GPU ", default=False)
        (options, args) = parser.parse_args()

        # Analyse aruments and options
        if options.verbose:
            logger.setLevel(logging.DEBUG)
        self.hdf5 = options.outfile
        if options.dark:
            darkFiles = [f for f in options.dark.split(",") if os.path.isfile(f)]
            if darkFiles:
                self.dark = fabio.open(darkFiles[0]).data.astype(numpy.float32)
                if len(darkFiles) > 1:
                    for i in darkFiles[1:]:
                        self.dark += fabio.open(i).data
                    self.dark /= len(darkFiles)
        if options.flat:
            flatFiles = [f for f in options.flat.split(",") if os.path.isfile(f)]
            if flatFiles:
                self.flat = fabio.open(flatFiles[0]).data.astype(numpy.float32)
                if len(flatFiles) > 1:
                    for i in flatFiles[1:]:
                        self.flat += fabio.open(i).data
                    self.flat /= len(flatFiles)
        self.use_gpu = options.gpu
        self.inputfiles = []
        for f in args:
            if os.path.isfile(f) and f.endswith(options.extension):
                self.inputfiles.append(f)
            elif os.path.isdir(f):
                self.inputfiles += [os.path.join(f, g) for g in os.listdir(f) if g.endswith(options.extention)]
        self.inputfiles.sort()
        if not self.inputfiles:
            raise RuntimeError("No input files to process")
        if options.poni:
            if os.path.isfile(options.poni):
                self.poni = options.poni
                self.setup_ai()
            else:
                logger.warning("No such poni file %s" % options.poni)
        if options.mask:
            if os.path.isfile(options.poni):
                self.mask = fabio.open(options.mask).data
            else:
                logger.warning("No such mask file %s" % options.poni)
        if options.nTrans is not None:
            self.nTrans = int(options.nTrans)
        if options.nRot is not None:
            self.nRot = int(options.nRot)
        if options.nDiff is not None:
            self.nDiff = int(options.nDiff)
        if options.offset is not None:
            self.offset = int(options.offset)

    def makeHDF5(self, rewrite=True):
        """
        Create the HDF5 structure if needed ...
        """
        if os.path.exists(self.hdf5) and rewrite:
            os.unlink(self.hdf5)
        h = h5py.File(self.hdf5)
        self.group = h.require_group(posixpath.dirname(self.hdf5path))

        if posixpath.basename(self.hdf5path) in self.group:
            self.dataset = self.group[posixpath.basename(self.hdf5path)]
        else:
            self.dataset = self.group.create_dataset(name=posixpath.basename(self.hdf5path),
                               shape=(self.nRot, self.nTrans, self.nDiff),
                               dtype="float32",
                               chunks=(1, self.nTrans, self.nDiff),
                               maxshape=(None, None, self.nDiff))
    def setup_ai(self):
        if self.poni:
            self.ai = pyFAI.load(self.poni)
        else:
            logger.error("Unable to setup Azimuthal integrator: no poni file provided")
            raise RuntimeError("You must provide poni a file")
        if self.dark is not None:
            self.ai.darkcurrent = self.dark
        if self.flat is not None:
            self.ai.flatfield = self.flat
        if self.mask is not None:
            self.ai.detector.mask = self.mask.astype("int8")

    def show_stats(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Unable to start matplotlib for display")
            return

        n, bins, patches = plt.hist(self.timing, 500, facecolor='green', alpha=0.75)
        plt.xlabel('Execution time in sec')
        plt.title("Execution time")
        plt.grid(True)
        plt.show()

    def get_pos(self, filename):
        """
        Calculate the position in the sinogram of the file according to it's number
        """
        n = int(filename.split(".")[0].split("_")[-1]) - (self.offset or 0)
        return {"index":n, "rot":n // self.nTrans, "trans": n % self.nTrans}

    def process_one_file(self, filename):
        """
        """
        if self.dataset is None:
            self.makeHDF5()
        if self.ai is None:
            self.setup_ai()
        t = time.time()
        pos = self.get_pos(filename)
        shape = self.dataset.shape
        if  pos["rot"] + 1 > shape[0]:
            self.dataset.resize((pos["rot"] + 1, shape[1], shape[2]))
        elif pos["index"] < 0 or pos["rot"] < 0 or pos["trans"] < 0:
            return
        data = fabio.open(filename).data.astype(numpy.float32)
        if self.use_gpu:
            tth, I = self.ai.xrpd_LUT_OCL(data, self.nDiff, safe=False, devicetype="gpu")
        else:
            tth, I = self.ai.xrpd_LUT(data, self.nDiff, safe=False)
        self.dataset[pos["rot"], pos["trans"], :] = I
        if "2theta" not in self.group:
            self.group["2theta"] = tth
        t -= time.time()
        print("Processing %30s took %6.1fms" % (os.path.basename(filename), -1000 * t))
        self.timing.append(-t)

    def process(self):
        if self.dataset is None:
            self.makeHDF5()
        t0 = time.time()
        for f in self.inputfiles:
            self.process_one_file(f)
        tot = time.time() - t0
        cnt = len(self.timing)
        print("Execution time for %i frames: %.3fs; Average execution time: %.1fms" % (cnt, tot, 1000.*tot / cnt))


if __name__ == "__main__":
    dt = DiffTomo()
    dt.parse()
    dt.makeHDF5()
    dt.process()
    dt.show_stats()


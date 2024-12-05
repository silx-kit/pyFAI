from pyFAI.units import Unit, get_unit_fiber, to_unit
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI.calibrant import get_calibrant
from pyFAI.integrator.fiber import FiberIntegrator
from pyFAI import detector_factory
import time
from pyFAI.gui.jupyter import subplots, display, plot2d
import matplotlib.pyplot as plt
import numpy
from pyFAI.test.utilstest import UtilsTest
import fabio
from pyFAI import load

if __name__ == "__main__":
    fi_1 = FiberIntegrator(dist=0.1, poni1=0.1, poni2=0.1, detector=detector_factory("Eiger_4M"), wavelength=1e-10)
    poni = UtilsTest.getimage("LaB6_5.poni")    
    fi_2 = load(filename=poni, type_="pyFAI.integrator.fiber.FiberIntegrator")
    
    cal = get_calibrant("LaB6")
    data_1 = cal.fake_calibration_image(ai=fi_1)
    data_file = UtilsTest.getimage("Y6.edf")
    data_2 = fabio.open(data_file).data

    for fi, data in zip((fi_1, fi_2), (data_1, data_2)):
        res2d_1 = fi.integrate2d_grazing_incidence(data=data, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1)
        res2d_2 = fi.integrate2d_grazing_incidence(data=data, incident_angle=0.2, tilt_angle=0.0, sample_orientation=1)
        res2d_3 = fi.integrate2d_grazing_incidence(data=data, incident_angle=0.0, tilt_angle=-0.2, sample_orientation=1)
        res2d_4 = fi.integrate2d_grazing_incidence(data=data, incident_angle=0.2, tilt_angle=-1.54, sample_orientation=1)

        res2d_5 = fi.integrate2d_grazing_incidence(data=data, incident_angle=0.0, tilt_angle=0.0, sample_orientation=2)
        res2d_6 = fi.integrate2d_grazing_incidence(data=data, incident_angle=0.0, tilt_angle=0.0, sample_orientation=3)
        res2d_7 = fi.integrate2d_grazing_incidence(data=data, incident_angle=0.0, tilt_angle=0.0, sample_orientation=4)

        fig, axes = subplots(ncols=4, nrows=2)
        plot2d(res2d_1, ax=axes[0,0])
        plot2d(res2d_2, ax=axes[0,1])
        plot2d(res2d_3, ax=axes[0,2])
        plot2d(res2d_4, ax=axes[0,3])
        plot2d(res2d_5, ax=axes[1,0])
        plot2d(res2d_6, ax=axes[1,1])
        plot2d(res2d_7, ax=axes[1,2])
        for ax in axes.ravel():
            if len(ax.get_images()) == 0:
                continue
            ax.get_images()[0].set_cmap("viridis")
            # ax.get_images()[0].set_clim(0,1500)
        plt.show()

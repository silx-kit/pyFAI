from pyFAI.gui.jupyter import plot1d, plot2d, subplots
import matplotlib.pyplot as plt
from pyFAI.test.utilstest import UtilsTest
from pyFAI import load
import fabio
from pyFAI import detector_factory
from pyFAI.calibrant import get_calibrant
from pyFAI.integrator.fiber import FiberIntegrator

if __name__ == "__main__":


    # EXAMPLE WITH CLASSIC QIP-QOOP NM^-1
    fi = load(UtilsTest.getimage("LaB6_3.poni"), type_="pyFAI.integrator.fiber.FiberIntegrator")
    data = fabio.open(UtilsTest.getimage("Y6.edf")).data
    air = fabio.open(UtilsTest.getimage("air.edf")).data
    data_clean = data - 0.03 * air
    sample_orientation = 6

    res2d_0 = fi.integrate2d_grazing_incidence(data=data_clean, sample_orientation=sample_orientation,
                                            )

    res2d_patch_1 = fi.integrate2d_grazing_incidence(data=data_clean, sample_orientation=sample_orientation,
                                            ip_range=[-1,1], oop_range=[0,20],
                                            )
    res2d_patch_2 = fi.integrate2d_grazing_incidence(data=data_clean, sample_orientation=sample_orientation,
                                            ip_range=[-17.5,17.5], oop_range=[5,6.5],
                                            )


    res1d_0 = fi.integrate1d_grazing_incidence(data=data_clean, sample_orientation=sample_orientation,
                                            ip_range=[-1,1], oop_range=[0,20],
                                            npt_ip=100, npt_oop=500,
                                            vertical_integration=True,
                                            )
    res1d_1 = fi.integrate1d_grazing_incidence(data=data_clean, sample_orientation=sample_orientation,
                                            ip_range=[-17.5,17.5], oop_range=[5,6.5],
                                            npt_ip=500, npt_oop=100,
                                            vertical_integration=False,
                                            )

    fig, ax = subplots(nrows=4, ncols=4)
    plot2d(result=res2d_0, ax=ax[0,0])
    img = ax[0,0].get_images()[0]
    img.set_cmap("viridis")
    img.set_clim(20,200)

    plot2d(result=res2d_patch_1, ax=ax[0,1])
    plot2d(result=res2d_patch_2, ax=ax[0,1])
    plot2d(result=res2d_0, ax=ax[0,1])

    ax[0,1].get_images()[0].set_cmap("viridis")
    ax[0,1].get_images()[1].set_cmap("viridis")
    ax[0,1].get_images()[2].set_cmap("gray")
    ax[0,1].get_images()[2].set_alpha(0.5)

    plot1d(result=res1d_0, ax=ax[0,2])
    plot1d(result=res1d_1, ax=ax[0,3])



    # EXAMPLE WITH POLAR UNITS RAD
    det = detector_factory("Eiger_4M")
    cal = get_calibrant("LaB6")
    fi = FiberIntegrator(detector=det, wavelength=1e-10, dist=0.1, poni1=0.05, poni2=0.05)
    data = cal.fake_calibration_image(ai=fi) * 1000 + 100

    sample_orientation = 1

    res2d= fi.integrate2d_grazing_incidence(data=data, sample_orientation=sample_orientation,
                                            )

    res2d_polar = fi.integrate2d_grazing_incidence(data=data, sample_orientation=sample_orientation,
                                                   unit_ip="qtot_nm^-1", unit_oop="chigi_rad",
                                            )
    res1d_0 = fi.integrate1d_grazing_incidence(data=data, sample_orientation=sample_orientation,
                                               unit_ip="qtot_nm^-1", unit_oop="chigi_rad",
                                            ip_range=[0,40], oop_range=[-0.2,0.2],
                                            npt_ip=1000, npt_oop=500,
                                            vertical_integration=False,
                                            )

    res2d_patch = fi.integrate2d_grazing_incidence(data=data, sample_orientation=sample_orientation,
                                               unit_ip="qtot_nm^-1", unit_oop="chigi_rad",
                                            ip_range=[0,40], oop_range=[-0.2,0.2],
                                            )

    plot2d(result=res2d, ax=ax[1,0])
    plot2d(result=res2d_polar, ax=ax[1,1])
    plot2d(result=res2d_patch, ax=ax[1,2])
    plot2d(result=res2d_polar, ax=ax[1,2])

    img = ax[1,0].get_images()[0]
    img.set_cmap("viridis")
    img = ax[1,1].get_images()[0]
    img.set_cmap("viridis")

    ax[1,2].get_images()[0].set_cmap("viridis")
    ax[1,2].get_images()[1].set_cmap("gray")
    ax[1,2].get_images()[1].set_alpha(0.5)
    plot1d(result=res1d_0, ax=ax[1,3])



    # EXAMPLE WITH POLAR UNITS DEG-QA-1
    det = detector_factory("Eiger_4M")
    cal = get_calibrant("LaB6")
    fi = FiberIntegrator(detector=det, wavelength=1e-10, dist=0.1, poni1=0.05, poni2=0.05)
    data = cal.fake_calibration_image(ai=fi) * 1000 + 100

    sample_orientation = 1

    res2d= fi.integrate2d_grazing_incidence(data=data, sample_orientation=sample_orientation,
                                            )

    res2d_polar = fi.integrate2d_grazing_incidence(data=data, sample_orientation=sample_orientation,
                                                   unit_ip="qtot_A^-1", unit_oop="chigi_deg",
                                            )
    res1d_0 = fi.integrate1d_grazing_incidence(data=data, sample_orientation=sample_orientation,
                                               unit_ip="qtot_A^-1", unit_oop="chigi_deg",
                                            ip_range=[0,4], oop_range=[-10,10],
                                            npt_ip=1000, npt_oop=500,
                                            vertical_integration=False,
                                            )

    res2d_patch = fi.integrate2d_grazing_incidence(data=data, sample_orientation=sample_orientation,
                                               unit_ip="qtot_A^-1", unit_oop="chigi_deg",
                                            ip_range=[0,4], oop_range=[-10,10],
                                            )

    plot2d(result=res2d, ax=ax[2,0])
    plot2d(result=res2d_polar, ax=ax[2,1])
    plot2d(result=res2d_patch, ax=ax[2,2])
    plot2d(result=res2d_polar, ax=ax[2,2])

    img = ax[2,0].get_images()[0]
    img.set_cmap("viridis")
    img = ax[2,1].get_images()[0]
    img.set_cmap("viridis")

    ax[2,2].get_images()[0].set_cmap("viridis")
    ax[2,2].get_images()[1].set_cmap("gray")
    ax[2,2].get_images()[1].set_alpha(0.5)
    plot1d(result=res1d_0, ax=ax[2,3])




    # EXAMPLE WITH POLAR UNITS DEG-QA-1 NOW VERTICAL
    det = detector_factory("Eiger_4M")
    cal = get_calibrant("LaB6")
    fi = FiberIntegrator(detector=det, wavelength=1e-10, dist=0.1, poni1=0.05, poni2=0.05)
    data = cal.fake_calibration_image(ai=fi) * 1000 + 100

    sample_orientation = 1

    res2d= fi.integrate2d_grazing_incidence(data=data, sample_orientation=sample_orientation,
                                            )

    res2d_polar = fi.integrate2d_grazing_incidence(data=data, sample_orientation=sample_orientation,
                                               unit_oop="qtot_A^-1", unit_ip="chigi_deg",
                                            )
    res1d_0 = fi.integrate1d_grazing_incidence(data=data, sample_orientation=sample_orientation,
                                               unit_oop="qtot_A^-1", unit_ip="chigi_deg",
                                            oop_range=[0,4], ip_range=[-10,10],
                                            npt_oop=1000, npt_ip=500,
                                            vertical_integration=True,
                                            )

    res2d_patch = fi.integrate2d_grazing_incidence(data=data, sample_orientation=sample_orientation,
                                               unit_oop="qtot_A^-1", unit_ip="chigi_deg",
                                            oop_range=[0,4], ip_range=[-10,10],
                                            )

    plot2d(result=res2d, ax=ax[3,0])
    plot2d(result=res2d_polar, ax=ax[3,1])
    plot2d(result=res2d_patch, ax=ax[3,2])
    plot2d(result=res2d_polar, ax=ax[3,2])

    img = ax[3,0].get_images()[0]
    img.set_cmap("viridis")
    img = ax[3,1].get_images()[0]
    img.set_cmap("viridis")

    ax[3,2].get_images()[0].set_cmap("viridis")
    ax[3,2].get_images()[1].set_cmap("gray")
    ax[3,2].get_images()[1].set_alpha(0.5)
    plot1d(result=res1d_0, ax=ax[3,3])

    plt.show()

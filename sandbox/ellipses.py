#!/usr/bin/env python3

import numpy
from math import sqrt, atan2, pi
from collections import namedtuple
from numpy import linspace, sin, cos, rad2deg

Ellipse = namedtuple("Ellipse", ["center_1", "center_2", "angle", "half_long_axis", "half_short_axis"])


def fit_ellipse(pty, ptx, _allow_delta=True):
    """Fit an ellipse

    inspired from
    http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

    :param pty: point coordinates in the slow dimension (y)
    :param ptx: point coordinates in the fast dimension (x)
    :raise ValueError: If the ellipse can't be fitted
    """
    x = ptx[:, numpy.newaxis]
    y = pty[:, numpy.newaxis]
    D = numpy.hstack((x * x, x * y, y * y, x, y, numpy.ones_like(x)))
    S = numpy.dot(D.T, D)
    try:
        inv = numpy.linalg.inv(S)
    except numpy.linalg.LinAlgError:
        if not _allow_delta:
            raise ValueError("Ellipse can't be fitted: singular matrix")
        # Try to do the same with a delta
        delta = 100
        ellipse = fit_ellipse(pty + delta, ptx + delta, _allow_delta=False)
        y0, x0, angle, wlong, wshort = ellipse
        return Ellipse(y0 - delta, x0 - delta, angle, wlong, wshort)

    C = numpy.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = numpy.linalg.eig(numpy.dot(inv, C))

    # First of all, sieve out all infinite and complex eigenvalues and come back to the Real world
    m = numpy.logical_and(numpy.isfinite(E), numpy.isreal(E))
    E, V = E[m].real, V[:, m].real

    # Ensures a>0, invert eigenvectors concerned
    V[:, V[0] < 0] = -V[:, V[0] < 0]
    # See https://mathworld.wolfram.com/Ellipse.html #15
    # Eigenvector must meet constraint (ac - b^2)>0 to be valid.
    A = V[0]
    B = V[1] / 2.0
    C = V[2]
    D = V[3] / 2.0
    F = V[4] / 2.0
    G = V[5]

    # Condition 1: Delta = det((a b d)(b c f)(d f g)) !=0
    Delta = A * (C * G - F * F) - G * B * B + D * (2 * B * F - C * D)
    # Condition 2: J>0
    J = (A * C - B * B)

    # Condition 3: Delta/(A+C)<0, replaces by Delta*(A+C)<0, less warnings
    m = numpy.logical_and(J > 0, Delta != 0)
    m = numpy.logical_and(m, Delta * (A + C) < 0)

    n = numpy.where(m)[0]
    if len(n) == 0:
        raise ValueError("Ellipse can't be fitted: No Eigenvalue match all 3 critera")
    else:
        n = n[0]
    a = A[n]
    b = B[n]
    c = C[n]
    d = D[n]
    f = F[n]
    g = G[n]
#     print(f"a {a}, b {b}, c {c}, ac-b² {a*c - b*b}")

    # Calculation of the center:
    denom = b * b - a * c
    x0 = (c * d - b * f) / denom
    y0 = (a * f - b * d) / denom

    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
#     print(f"up {up}, down1 {down1}, down2 {down2}")
    a2 = up / down1
    b2 = up / down2
    if a2 <= 0 or b2 <= 0:
        raise ValueError("Ellipse can't be fitted, negative sqrt")

    res1 = sqrt(a2)
    res2 = sqrt(b2)

    if a == c:
        angle = 0  # we have a circle
    elif res2 > res1:
        res1, res2 = res2, res1
        angle = 0.5 * (pi + atan2(2 * b, (a - c)))
    else:
        angle = 0.5 * (pi + atan2(2 * b, (a - c)))
    return Ellipse(y0, x0, angle, res1, res2)


def rotate(ptx, pty, angle=0):
    v = numpy.vstack((ptx, pty))
    ra = numpy.deg2rad(angle)
    rot = numpy.array([[cos(ra), -sin(ra)], [sin(ra), cos(ra)]])
    resx, resy = rot.dot(v)
    return resx, resy


def display(ptx, pty, ellipse=None, ax=None):
    if ax is None:
        fig, ax = subplots()
    if ellipse is not None:
        t = numpy.linspace(0, 2 * pi, 1000, endpoint=False)
        x = ellipse.half_long_axis * cos(t)
        y = ellipse.half_short_axis * sin(t)
        x, y = rotate(x, y, numpy.rad2deg(ellipse.angle))
        x += ellipse.center_2
        y += ellipse.center_1
        ax.plot(x, y, ",g")
        ylim = min(y.min(), pty.min()), max(y.max(), pty.max())
        xlim = min(x.min(), ptx.min()), max(x.max(), ptx.max())

    else:
        ylim = pty.min(), pty.max()
        xlim = ptx.min(), ptx.max()
    ax.plot(ptx, pty, "ro", color="blue")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return ax


def display_one(x0, y0, phi, a, b, npt=10, extent=1.0, ax=None):
    if ax is None:
        fig, ax = subplots()

    print("*"*50)
    print(Ellipse(y0, x0, numpy.deg2rad(phi), a, b))
    angles = linspace(0, 2 * pi * extent, npt, endpoint=False)
    ptx = cos(angles) * a
    pty = sin(angles) * b
    ptx, pty = rotate(ptx, pty, phi)
    ptx += x0
    pty += y0

    try:
        ellipse = fit_ellipse(pty, ptx)
    except Exception as e:
        print(e)
        ellipse = None
    display(ptx, pty, ellipse, ax=ax)
    print(ellipse)
    return ax


def test_one(ellipse, npt=10, extent=1):
    ref = ellipse

    angles = linspace(0, 2 * pi * extent, npt, endpoint=False)
    ["center_1", "center_2", "angle", "half_long_axis", "half_short_axis"]
    ptx = cos(angles) * ellipse.half_long_axis
    pty = sin(angles) * ellipse.half_short_axis
    ptx, pty = rotate(ptx, pty, numpy.rad2deg(ellipse.angle))
    ptx += ellipse.center_2
    pty += ellipse.center_1

    try:
        obt = fit_ellipse(pty, ptx)
    except Exception as e:
        print(e)
        print(f"Failed with {ref}")
        return False
    if abs(obt.center_1 - ref.center_1) > 1:
        print("Center_1:\n", ref, "\n", obt)
        return False
    if abs(obt.center_2 - ref.center_2) > 1:
        print("Center_2:\n", ref, "\n", obt)
        return False
    if abs(obt.half_long_axis - ref.half_long_axis) > 1:
        print("half_long_axis:\n", ref, "\n", obt)
        return False
    if abs(obt.half_short_axis - ref.half_short_axis) > 1:
        print("half_short_axis:\n", ref, "\n", obt)
        return False
    if abs(rad2deg(obt.angle % pi) - rad2deg(ref.angle % pi)) > 5:
        if ref.half_long_axis / obt.half_long_axis > 1.1:
            print("angle:\n", ref, "\n", obt)
            return False
    return True


if __name__ == "__main__":
    from matplotlib.pyplot import subplots
    # from matplotlib import patches
    fig, ax = subplots(4, 4)
    display_one(x0=5, y0=0, phi=0, a=15, b=10, npt=9, ax=ax[0, 0])
    display_one(x0=5, y0=0, phi=0, a=15, b=10, npt=10, ax=ax[1, 0])
    display_one(x0=5, y0=0, phi=0, a=15, b=10, npt=11, ax=ax[2, 0])
    display_one(x0=5, y0=0, phi=0, a=15, b=10, npt=12, ax=ax[3, 0])

    display_one(x0=5, y0=-5, phi=10, a=15, b=10, npt=9, ax=ax[0, 1])
    display_one(x0=5, y0=-5, phi=10, a=15, b=10, npt=10, ax=ax[1, 1])
    display_one(x0=5, y0=-5, phi=10, a=15, b=10, npt=11, ax=ax[2, 1])
    display_one(x0=5, y0=-5, phi=10, a=15, b=10, npt=12, ax=ax[3, 1])

    display_one(x0=5, y0=-5, phi=-10, a=15, b=10, npt=9, ax=ax[0, 2])
    display_one(x0=5, y0=-5, phi=-10, a=15, b=10, npt=10, ax=ax[1, 2])
    display_one(x0=5, y0=-5, phi=-10, a=15, b=10, npt=11, ax=ax[2, 2])
    display_one(x0=5, y0=-5, phi=-10, a=15, b=10, npt=12, ax=ax[3, 2])

    display_one(x0=5, y0=-5, phi=90, a=15, b=10, npt=9, ax=ax[0, 3])
    display_one(x0=5, y0=-5, phi=90, a=15, b=10, npt=10, ax=ax[1, 3])
    display_one(x0=5, y0=-5, phi=90, a=15, b=10, npt=11, ax=ax[2, 3])
    display_one(x0=5, y0=-5, phi=90, a=15, b=10, npt=12, ax=ax[3, 3])
    fig.show()
    fig2, ax = subplots(4, 4)
    import random
    for i in range(16):
        center_x = random.randint(-10, 10)
        center_y = random.randint(-10, 10)
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        a, b = max(b, a), min(b, a)
        angle = random.randint(-90, 90)
        npt = random.randint(10, 30)
        extent = random.random() + 0.2
        display_one(x0=center_x, y0=center_y, phi=angle, a=a, b=b, npt=npt, extent=extent, ax=ax[i // 4, i % 4])
        fig2.show()

    print("#"*50)
    i = 0
    OK = True
    while OK and i < 1000:
        center_x = random.randint(-10, 10)
        center_y = random.randint(-10, 10)
        a = random.randint(1, 20)
        b = random.randint(1, 20)
        a, b = max(b, a), min(b, a)
        angle = random.randint(-360, 360)
        ellipse = Ellipse(center_y, center_x, numpy.deg2rad(angle), a, b)
        OK = test_one(ellipse, npt=random.randint(10, 30), extent=random.random() + 0.2)
        i += 1
    print(f"Tested {i} configuration without issue")
    input("enter to quit")


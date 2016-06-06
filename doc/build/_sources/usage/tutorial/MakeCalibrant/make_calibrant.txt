
Creation of a calibrant file
============================

In this tutorial we will see how to generate a file describing a
*reference material* used as calibrant to refine the geometry of the
experimental setup, especially the position of the detector.

In pyFAI, calibrant are defined in a bunch of files available from
`github <https://github.com/kif/pyFAI/tree/master/calibration>`__. Those
files are automatically installed with pyFAI and readily available from
the programming interface, as described in the **Calibrant** tutorial.

This tutorials focuses on the content of the file and how to generate
other calibrant files and how to exended existing ones.

Content of the file
-------------------

Calibrant files from *pyFAI* are heavily inspired from the ones used in
*Fit2D*: they simply store the *d-spacing* between Miller plans and are
simply loaded using the *numpy.loadtxt* function. In pyFAI we have
improved with plent of comment (using **#**) to provide in addition the
familly of Miller plan associated and the multiplicity, which, while not
proportionnal to *Fobs* is somewhat related.

One may think it would have been better/simpler to describe the cell of
the material. Actually some calibrant, especially the ones used in SAXS
like silver behenate (AgBh), but not only, are not well crystallized
compounds and providing the *d-spacing* allows to use them as calibrant.
Neverthless this tutorial will be focused on how to generate calibrant
files fron crystal structures.

Finally the *calibrant file* has a few lines of headers containing the
name, the crystal cell parameters (if available) and quite importantly a
reference to the source of information like a publication to allow the
re-generation of the file if needed.

::

    # Calibrant: Silver Behenate (AgBh)
    # Pseudocrystal a=inf b=inf c=58.380
    # Ref: doi:10.1107/S0021889899012388
    5.83800000e+01 # (0,0,1)
    2.91900000e+01 # (0,0,2)
    1.94600000e+01 # (0,0,3)
    1.45950000e+01 # (0,0,4)
    1.16760000e+01 # (0,0,5)
    9.73000000e+00 # (0,0,6)
    8.34000000e+00 # (0,0,7)
    7.29750000e+00 # (0,0,8)
    6.48666667e+00 # (0,0,9)
    (...)

The Cell class
--------------

To generate a *calibrant file* from a crystal structure, the easiest is
to use the **pyFAI.calibrant.Cell** class.

.. code:: python

    from pyFAI.calibrant import Cell
    print(Cell.__doc__)
    print(Cell.__init__.__doc__)


.. parsed-literal::

    
        This is a cell object, able to calculate the volume and d-spacing according to formula from:
    
        http://geoweb3.princeton.edu/research/MineralPhy/xtalgeometry.pdf
        
    Constructor of the Cell class:
    
            Crystalographic units are Angstrom for distances and degrees for angles !
    
            @param a,b,c: unit cell length in Angstrom
            @param alpha, beta, gamma: unit cell angle in degrees
            @param lattice: "cubic", "tetragonal", "hexagonal", "rhombohedral", "orthorhombic", "monoclinic", "triclinic"
            @param lattice_type: P, I, F, C or R
            


The constructor of the class is used to build and well suited to
triclinic crystal.

Specific constructors
~~~~~~~~~~~~~~~~~~~~~

| Nevertheless, most used calibrants are of much higher symmetry, like
cubic which takes only **one** parameter.
| Here is an example for defining
`Polonium <http://www.periodictable.com/Elements/084/data.html>`__ which
is a simple cubic cell (Primitive) with a cell parameter of 335.9pm.
This example was chosen as Polonium is apparently the only element with
such primitive cubic packing.

.. code:: python

    Po = Cell.cubic(3.359)
    print(Po)

.. parsed-literal::

    Primitive cubic cell a=3.3590 b=3.3590 c=3.3590 alpha=90.000 beta=90.000 gamma=90.000


.. code:: python

    print(Po.volume)

.. parsed-literal::

    37.899197279


.. code:: python

    print("Po.d_spacing?")
    print(Po.d_spacing.__doc__)
    print("Po.save?")
    print(Po.save.__doc__)

.. parsed-literal::

    Po.d_spacing?
    Calculate all d-spacing down to dmin
    
            applies selection rules
    
            @param dmin: minimum value of spacing requested
            @return: dict d-spacing as string, list of tuple with Miller indices
                    preceded with the numerical value
            
    Po.save?
    Save informations about the cell in a d-spacing file, usable as Calibrant
    
            @param name: name of the calibrant
            @param doi: reference of the publication used to parametrize the cell
            @param dmin: minimal d-spacing
            @param dest_dir: name of the directory where to save the result
            


To generate a *d-spacing* file usable as calibrant, one simply has to
call the *save* method of the *Cell* instance.

**Nota:** the ".D" suffix is automatically appended.

.. code:: python

    Po.save("Po",doi="http://www.periodictable.com/Elements/084/data.html", dmin=1)
.. code:: python

    !cat Po.D

.. parsed-literal::

    # Calibrant: Po
    # Primitive cubic cell a=3.3590 b=3.3590 c=3.3590 alpha=90.000 beta=90.000 gamma=90.000
    # Ref: http://www.periodictable.com/Elements/084/data.html
    3.35900000 # (1, 0, 0) 6
    2.37517168 # (1, 1, 0) 12
    1.93931955 # (1, 1, 1) 8
    1.67950000 # (2, 0, 0) 6
    1.50219047 # (2, 1, 0) 24
    1.37130601 # (2, 1, 1) 24
    1.18758584 # (2, 2, 0) 12
    1.11966667 # (3, 0, 0) 30
    1.06220907 # (3, 1, 0) 24
    1.01277661 # (3, 1, 1) 24


Other Examples: LaB6
~~~~~~~~~~~~~~~~~~~~

Lantanide Hexaboride, probably my favorite calibrant as it has a
primitive cubic cell hence all reflections are valid and intense. The
cell parameter is available from the
`NIST <https://www-s.nist.gov/srmors/certificates/660C.pdf>`__ at
a=4.156826.

The number of reflections in a file is controled by the *dmin*
parameter. The lower it is, the more lines there are and the more
time-consuming this will be:

.. code:: python

    LaB6 = Cell.cubic(4.156826)
    %time d=LaB6.d_spacing(0.1)
    print("Number of reflections: %s"%len(d))

.. parsed-literal::

    CPU times: user 976 ms, sys: 76 ms, total: 1.05 s
    Wall time: 964 ms
    Number of reflections: 1441


.. code:: python

    LaB6.save("LaB6",doi="https://www-s.nist.gov/srmors/certificates/660C.pdf", dmin=0.1)
.. code:: python

    !head LaB6.D

.. parsed-literal::

    # Calibrant: LaB6
    # Primitive cubic cell a=4.1568 b=4.1568 c=4.1568 alpha=90.000 beta=90.000 gamma=90.000
    # Ref: https://www-s.nist.gov/srmors/certificates/660C.pdf
    4.15682600 # (1, 0, 0) 6
    2.93931985 # (1, 1, 0) 12
    2.39994461 # (1, 1, 1) 8
    2.07841300 # (2, 0, 0) 6
    1.85898910 # (2, 1, 0) 24
    1.69701711 # (2, 1, 1) 24
    1.46965993 # (2, 2, 0) 12


Other Examples: Silicon
~~~~~~~~~~~~~~~~~~~~~~~

Silicon is easy to obtain **pure** thanks to microelectronics industry.
Its cell is a face centered cubic with a diamond like structure. The
cell parameter is available from the
`NIST <https://www-s.nist.gov/srmors/certificates/640E.pdf>`__:
a=5.431179 A.

Let's compare the difference between the silicon structure and the
equivalent FCC structure:

.. code:: python

    Si = Cell.diamond(5.431179)
    print(Si)

.. parsed-literal::

    Face centered cubic cell a=5.4312 b=5.4312 c=5.4312 alpha=90.000 beta=90.000 gamma=90.000


.. code:: python

    FCC = Cell.cubic(5.431179,"F")
    print(FCC)

.. parsed-literal::

    Face centered cubic cell a=5.4312 b=5.4312 c=5.4312 alpha=90.000 beta=90.000 gamma=90.000


Apparently, there is not difference. But to check it, let's generate all
lines down to 1A and compare them:

.. code:: python

    sid = Si.d_spacing(1)
    for key, val in sid.items():
        print("%s: %s"%(sorted(val[1:][-1]),key))

.. parsed-literal::

    [2, 2, 4]: 1.10863477e+00
    [1, 1, 3]: 1.63756208e+00
    [0, 2, 2]: 1.92021175e+00
    [1, 1, 1]: 3.13569266e+00
    [1, 1, 5]: 1.04523089e+00
    [1, 3, 3]: 1.24599792e+00
    [0, 0, 4]: 1.35779475e+00


.. code:: python

    fccd = FCC.d_spacing(1)
    for key, val in fccd.items():
        print("%s: %s"%(sorted(val[1:][-1]),key))

.. parsed-literal::

    [2, 2, 4]: 1.10863477e+00
    [1, 1, 3]: 1.63756208e+00
    [0, 2, 2]: 1.92021175e+00
    [1, 1, 1]: 3.13569266e+00
    [1, 1, 5]: 1.04523089e+00
    [0, 2, 4]: 1.21444854e+00
    [2, 2, 2]: 1.56784633e+00
    [1, 3, 3]: 1.24599792e+00
    [0, 0, 2]: 2.71558950e+00
    [0, 0, 4]: 1.35779475e+00


So there are many more reflection in the FCC structure compared to the
diamond-like structure: (4 2 0), (2 2 2) are extinct as h+k+l=4n and all
even.

Selection rules
~~~~~~~~~~~~~~~

Cell object contain *selection\_rules* which tells if a reflection is
allowed or not. Those *selection\_rules* can be introspected:

.. code:: python

    print(Si.selection_rules)
    print(FCC.selection_rules)

.. parsed-literal::

    [<function <lambda> at 0x7f8080fad848>, <function <lambda> at 0x7f8080fad938>, <function <lambda> at 0x7f8080fad9b0>]
    [<function <lambda> at 0x7f8080fadaa0>, <function <lambda> at 0x7f8080fada28>]


The *Si* object has one additionnal selection rule which explains the
difference: A specific rule telling that reflection with h+k+l=4n is
forbidden when (h,k,l) are all even.

We will now have a look at the source code of those selection rules
using the *inspect* module.

**Nota:** This is advanced Python hacking but useful for the
demonstration

.. code:: python

    import inspect
    for rule in Si.selection_rules: 
        print(inspect.getsource(rule))

.. parsed-literal::

            self.selection_rules = [lambda h, k, l: not(h == 0 and k == 0 and l == 0)]
    
                self.selection_rules.append(lambda h, k, l: (h % 2 + k % 2 + l % 2) in (0, 3))
    
                lambda h, k, l:not((h % 2 == 0) and (k % 2 == 0) and (l % 2 == 0) and ((h + k + l) % 4 != 0))
    


Actually the last line correspond to an anonymous function (lambda)
which implements this rule.

As we have seen previously one can simply adapt the Cell instance by
simply appending extra selection rules to this list.

A selection rule is a function taking 3 integers as input and returning
*True* if the reflection is allowed and *False* when it is forbidden by
symmetry. By this way one can simply adapt existing object to generate
the right *calibrant file*.

Conclusion
----------

In this tutorial we have seen the structure of a *calibrant file*, how
to generate crystal structure cell object to write such calibrant files
automatically, including all metadata needed for redistribution. Most
advanced programmers can now modify the selection rules to remove
forbidden reflection for a given cell.


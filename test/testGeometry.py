#!/usr/bin/env python

"""tests for Jon's geometry changes
FIXME : make some tests that the functions do what is expected

"""


import unittest, numpy
from pyFAI import geometry as geometry


class ParameterisedTestCase(unittest.TestCase):
    """ TestCase classes that want to be parameterised should
        inherit from this class.
        From Eli Bendersky's website
        http://eli.thegreenplace.net/2011/08/02/python-unit-testing-parametrized-test-cases/
    """
    def __init__(self, methodName='runTest', param=None):
        super(ParameterisedTestCase, self).__init__(methodName)
        self.param = param

    @staticmethod
    def parameterise(testcase_klass, param=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameter 'param'.
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_klass(name, param=param))
        return suite

class TestGeometry(ParameterisedTestCase):

    def testGeometryFunctions(self):
        oldfunc, newfunc, args, kwds, expectedFail = self.param
        g = geometry.Geometry( **kwds )
        oldret = getattr(g, oldfunc)(*args)
        newret = getattr(g, newfunc)(*args)
        nerr = numpy.testing.utils.nulp_diff( oldret, newret )
        msg = "%s %s %s %s"%(str(self.param),str(nerr),str(oldret),str(newret))
        if expectedFail:
            self.assertTrue( nerr > 20 , msg)
        else:
            self.assertFalse( nerr > 20 , msg)

TESTCASES = [
 ( "tth", "oldtth", (1,1), {'rot1':-1, 'rot2':1, 'rot3':1},False),
 ( "tth", "oldtth", (10,1), {'rot1':-.2, 'rot2':1, 'rot3':-.1},False),
 ( "tth", "oldtth", (1,10), {'rot1':-1, 'rot2':-.2, 'rot3':1},False),
 ( "tth", "oldtth", (10,10), {'rot1':1, 'rot2':5, 'rot3':.4},False),
 ( "tth", "oldtth", (2,10), {'rot1':-1.2, 'rot2':1, 'rot3':1},False),
 ( "tth", "oldtth", (1,1), {'dist':1e10, 'rot1':-1, 'rot2':1, 'rot3':1},False),
 ( "chi", "oldchi", (5,6) ,{},False),
 ( "chi", "oldchi", (-5,6) ,{},False),
 ( "chi", "oldchi", (-5,-6) ,{},False),
 ( "chi", "oldchi", (5,-6) ,{},False),
 ( "chi", "oldchi", (1,10), {'rot1':1},False),
 ( "chi", "oldchi", (2,1) , {'dist':2, 'rot2':1},False),
 ( "chi", "oldchi", (2,1) , {'rot2':-1},False),
 ( "chi", "oldchi", (2,1) , {'rot2':10},False),
 ( "chi", "oldchi", (-1,1), {'rot3':1},False),
 ( "chi", "oldchi", (-2,-3), {'rot1':1, 'rot2':2},False),
 ( "chi", "oldchi", (1,-1), {'rot1':0.5, 'rot2':1, 'rot3':5.9},False),
 ]
#trial( "qFunction", "rqFunction", (1,1), wavelength=1.0e-10 , dist=1e10)

# u-saxs, detector is far back. Issue with acos(1)
#trial( "tth", "newtth", (1,1), dist=1e9, True)




def test_suite_all_Geometry():
    testSuite = unittest.TestSuite()
    i=0
    for param in TESTCASES:
        testSuite.addTest( ParameterisedTestCase.parameterise( 
                TestGeometry, param) )
    return testSuite



if __name__ == '__main__':
    mysuite = test_suite_all_Geometry()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)

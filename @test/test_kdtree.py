#!/usr/bin/env python
import math
import unittest
import numpy as np
from kdtree import KDTree, KNNResultSet, RadiusResultSet

class Test_KDTree(unittest.TestCase):
    def setUp(self):
        pnts = np.array((
            (0.,0.,0.),
            (1.,0.,0.),
            (1.,1.,0.),
            (0.,1.,0.),
            (0.,0.,1.),
            (1.,0.,1.),
            (1.,1.,1.),
            (0.,1.,1.),
        ))
        
        self.tree1 = KDTree(pnts, maxLeafSize = 1)
        self.tree1.build()
        
        self.tree2 = KDTree(pnts, maxLeafSize = 1)
        self.tree2.build()
        
    def test_closest(self):
        idx, dist = self.tree1.closest((.1,.1,.1))
        self.assertEqual(idx, 0)
        self.assertAlmostEqual(dist, 3*.1**2)
        
        idx, dist = self.tree2.closest((.1,.1,.1))
        self.assertEqual(idx, 0)
        self.assertAlmostEqual(dist, 3*.1**2)
        
        idx, dist = self.tree1.closest((.9,.9,.9))
        self.assertEqual(idx, 6)
        self.assertAlmostEqual(dist, 3*.1**2)
        
        idx, dist = self.tree2.closest((.9,.9,.9))
        self.assertEqual(idx, 6)
        self.assertAlmostEqual(dist, 3*.1**2)
        
    def test_search(self):
        res = self.tree1.search((.5,.5,.5), .5)
        idx = set(res.ind)
        self.assertEqual(idx, set())
        
        res = self.tree2.search((.5,.5,.5), .5)
        idx = set(res.ind)
        self.assertEqual(idx, set())
        
        res = self.tree1.search((0.,0.,0.), .1)
        idx = set(res.ind)
        self.assertEqual(idx, set([0]))
        
        res = self.tree2.search((0.,0.,0.), .1)
        idx = set(res.ind)
        self.assertEqual(idx, set([0]))
        
        res = self.tree1.search((0.,0.,0.), 1.1)
        idx = set(res.ind)
        self.assertEqual(idx, set([0,1,3,4]))
        
        res = self.tree2.search((0.,0.,0.), 1.1)
        idx = set(res.ind)
        self.assertEqual(idx, set([0,1,3,4]))
    
    def test_searchBBox(self):
        res = self.tree1.searchBBox((.25,.25,.25), (.75,.75,.75))
        idx = set(res.ind)
        self.assertEqual(idx, set())
        
        res = self.tree2.searchBBox((.25,.25,.25), (.75,.75,.75))
        idx = set(res.ind)
        self.assertEqual(idx, set())
        
        res = self.tree1.searchBBox((0.,0.,0.), (1.,1.,1.))
        idx = set(res.ind)
        self.assertEqual(idx, set(range(8)))
        
        res = self.tree2.searchBBox((0.,0.,0.), (1.,1.,1.))
        idx = set(res.ind)
        self.assertEqual(idx, set(range(8)))
        
        res = self.tree1.searchBBox((-.1,-.1,-1.), (.1,.1,.1))
        idx = set(res.ind)
        self.assertEqual(idx, set([0]))
        
        res = self.tree2.searchBBox((-.1,-.1,-1.), (.1,.1,.1))
        idx = set(res.ind)
        self.assertEqual(idx, set([0]))
        
        
if __name__ == '__main__':
    unittest.main(verbosity=2)
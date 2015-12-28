# -*- coding: utf-8 -*-
#
# Kd-Tree
#
import numpy as np

class Node:
    pass

class KNNResultSet:
    def __init__(self, capacity):
        self.capacity = capacity
        self.count = 0
        
        self.indices = np.zeros((capacity,), np.uint32)
        self.dists = np.zeros((capacity,), np.float64)
        
        # ensure we have first match
        self.dists[capacity-1] = (1<<32) - 1
    
    def __repr__(self):
        args = self.capacity, self.count
        return 'KNNResultSet(capacity=%d, count=%d)' % args
    
    __str__ = __repr__
    
    def __len__(self):
        '''
        Return length of result set
        '''
        return self.count
    
    def __bool__(self):
        '''
        Test if result set contains any data
        '''
        return bool(self.count)
    
    __hash__ = None
    
    def __iter__(self):
        '''
        Iterate over result
        '''
        for i in range(self.count):
            yield self.indices[i], self.dists[i]
    
    def __reversed__(self):
        '''
        Reversed iteration over values in Bitset
        '''
        for i in reversed(range(self.count)):
            yield self.indices[i], self.dists[i]
                
    def isFull(self):
        return self.count == self.capacity
    
    def addPoint(self, dist, index):
        i = self.count
        while i > 0:
            if self.dists[i-1] > dist:
                if i < self.capacity:
                    self.dists[i] = self.dists[i-1]
                    self.indices[i] = self.indices[i-1]
            else:
                break
            i -= 1
        
        if i < self.capacity:
            self.dists[i] = dist
            self.indices[i] = index
        
        if self.count < self.capacity:
            self.count += 1
    
    def worstDist(self):
        return self.dists[self.capacity - 1]
    
class RangeResultSet:
    def __init__(self):
        self.res = []
    
    def __repr__(self):
        return 'RangeResultSet(size=%d)' % len(self.res)
    
    __str__ = __repr__
    
    def __len__(self):
        '''
        Return length of result set
        '''
        return len(self.res)
    
    def __bool__(self):
        '''
        Test if result set contains any data
        '''
        return bool(self.res)
    
    __hash__ = None
    
    def __iter__(self):
        '''
        Iterate over result
        '''
        for index in self.res:
            yield index
    
    def __reversed__(self):
        '''
        Reversed iteration over values in Bitset
        '''
        for index in reversed(self.res):
            yield index
            
    def clear(self):
        self.res.clear()
    
    def size(self):
        return len(self.res)
    
    def isFull(self):
        return True
    
    def addPoint(self, index):
        self.res.append(index)
            
class RadiusResultSet:
    def __init__(self, radius):
        self.radius = radius
        self.res = []
    
    def __repr__(self):
        args = len(self.res), self.radius
        return 'RadiusResultSet(size=%d, radius=%g)' % args
    
    __str__ = __repr__
    
    def __len__(self):
        '''
        Return length of result set
        '''
        return len(self.res)
    
    def __bool__(self):
        '''
        Test if result set contains any data
        '''
        return bool(self.res)
    
    __hash__ = None
    
    def __iter__(self):
        '''
        Iterate over result
        '''
        for index in self.res:
            yield index
    
    def __reversed__(self):
        '''
        Reversed iteration over values in Bitset
        '''
        for index in reversed(self.res):
            yield index
            
    def clear(self):
        self.res.clear()
    
    def size(self):
        return len(self.res)
    
    def isFull(self):
        return True
    
    def addPoint(self, dist, index):
        if dist < self.radius:
            self.res.append((index, dist))
    
    def worstDist(self):
        return self.radius
            
class KDTree:
    '''
    KD search tree optimized for point clouds of 3d points.
    
    Contains the k-d trees and other information for indexing a set of points
    for nearest-neighbor matching.
    '''
    def __init__(self, points, bbox = None, maxLeafSize = 10):        
        self.pnts = np.asarray(points)
        
        self.npoints, dim = self.pnts.shape
        self.DIM = dim
        assert self.npoints > 0 and self.npoints < (1<<32) - 1
        
        if bbox is None:
            self.bbox = self.findBBox()
        else:
            self.bbox = np.asarray(bbox)
            assert self.bbox.shape == (2,self.DIM)
        
        self.maxLeafSize = maxLeafSize
        
        # Array of indices to vectors in the dataset.
        self.ind = np.arange(0, self.npoints, dtype = np.uint32)
        
        self.root = None
        
    
    def __repr__(self):
        args = self.npoints, self.DIM, not self.root is None
        return 'KDTree(npoints=%d, dim=%d, hasIndex=%s)' % args
    
    __str__ = __repr__
    
    def __len__(self):
        '''
        Return length of result set
        '''
        return self.npoints
    
    def __bool__(self):
        '''
        Test if result set contains any data
        '''
        return bool(self.npoints)
    
    __hash__ = None
    
    def findBBox(self):
        '''
        Calculate bounding box of points
        '''
        minv = self.pnts.min(axis = 0)
        maxv = self.pnts.max(axis = 0)
        return np.vstack((minv, maxv))
    
    def build(self):
        '''
        Recursive build the tree structure
        '''
        self.root = self.divideTree(0, self.npoints, self.bbox)
    
    def divideTree(self, left, right, bbox):
        # create node
        node = Node()
        
        # If too few exemplars remain, then make this a leaf node.
        if (right - left) <= self.maxLeafSize:
            # mark as leaf node
            node.child1 = node.child2 = None
            node.left = left
            node.right = right
            
            # compute bounding-box of leaf points
            for i in range(self.DIM):
                bbox[0,i] = self.pnts[self.ind[left], i]
                bbox[1,i] = self.pnts[self.ind[left], i]
            
            for i in range(left + 1, right):
                for j in range(self.DIM):
                    bbox[0,j] = min(bbox[0,j], self.pnts[self.ind[i], j])
                    bbox[1,j] = max(bbox[1,j], self.pnts[self.ind[i], j])
        else:
            idx, cutfeat, cutval = self.middleSplit(left, right - left, bbox)
            
            node.divfeat = cutfeat
            
            left_bbox = bbox.copy()
            left_bbox[1,cutfeat] = cutval
            node.child1 = self.divideTree(left, left + idx, left_bbox)
            
            right_bbox = bbox.copy()
            right_bbox[0,cutfeat] = cutval
            node.child2 = self.divideTree(left+idx, right, right_bbox)
            
            node.divlow = left_bbox[1, cutfeat]
            node.divhigh = right_bbox[0, cutfeat]
            
            for i in range(self.DIM):
                bbox[0, i] = min(left_bbox[0, i], right_bbox[0, i])
                bbox[1, i] = max(left_bbox[1, i], right_bbox[1, i])
            
        return node
    
    def computeMinMax(self, ind, count, element):
        min_element = self.pnts[self.ind[ind], element]
        max_element = self.pnts[self.ind[ind], element]
        
        for i in range(1, count):
            val = self.pnts[self.ind[ind + i], element]
            if val < min_element:
                min_element = val
            if val > max_element:
                max_element = val
        
        return min_element, max_element
        
    def middleSplit(self, ind, count, bbox):
        EPS = .00001
        max_span = bbox[1,0] - bbox[0,0]
        
        for i in range(1, self.DIM):
            span = bbox[1,i] - bbox[0,i]
            if span > max_span:
                max_span = span
        
        max_spread = -1
        cutfeat = 0
        for i in range(self.DIM):
            span = bbox[1,i] - bbox[0,i]
            if span > (1. - EPS)*max_span:
                min_element, max_element = self.computeMinMax(ind, count, cutfeat)
                spread = max_element - min_element
                if spread > max_spread:
                    cutfeat = i
                    max_spread = spread
        
        # split in the middle
        split_val = .5*(bbox[1,cutfeat] + bbox[0,cutfeat])
        min_elem, max_elem = self.computeMinMax(ind, count, cutfeat)
        
        if split_val < min_elem:
            cutval = min_elem
        elif split_val > max_elem:
            cutval = max_elem
        else:
            cutval = split_val
        
        lim1, lim2 = self.planeSplit(ind, count, cutfeat, cutval)
        
        if lim1 > count // 2:
            index = lim1
        elif lim2 < count // 2:
            index = lim2
        else:
            index = count // 2
        
        return index, cutfeat, cutval
    
    def planeSplit(self, ind, count, cutfeat, cutval):
        '''
        Subdivide the list of points by a plane perpendicular on axe corresponding
        to the 'cutfeat' dimension at 'cutval' position.
        '''
        left = 0
        right = count - 1
        
        # Move vector indices for left subtree to front of list.
        while True:
            while left <= right and self.pnts[self.ind[ind+left], cutfeat] < cutval:
                left += 1
            
            while right and left <= right and \
                   self.pnts[self.ind[ind+right], cutfeat] >= cutval:
                right -= 1
            
            if left > right or not right:
                break
            
            self.ind[ind+left], self.ind[ind+right] = self.ind[ind+right], self.ind[ind+left]
            left += 1
            right -= 1
        
        # If either list is empty, it means that all remaining features
        # are identical. Split in the middle to maintain a balanced tree.
        lim1 = left
        right = count - 1
        while True:
            while left <= right and self.pnts[self.ind[ind+left], cutfeat] <= cutval:
                left += 1
            
            while right and left <= right and \
                   self.pnts[self.ind[ind+right], cutfeat] > cutval:
                right -= 1
            
            if left > right or not right:
                break
            
            self.ind[ind+left], self.ind[ind+right] = self.ind[ind+right], self.ind[ind+left]
            left += 1
            right -= 1
        
        lim2 = left
        
        return lim1, lim2
    
    def computeInitialDistances(self, vec):
        dists = np.zeros((self.DIM,), np.float64)
        distsq = 0
        for i in range(self.DIM):
            if vec[i] < self.bbox[0, i]:
                dists[i] = self.accum_dist(vec[i], self.bbox[0, i])
                distsq += dists[i]
            if vec[i] > self.bbox[1, i]:
                dists[i] = self.accum_dist(vec[i], self.bbox[1, i])
                distsq += dists[i]
        
        return dists, distsq
    
    def findNeighbors(self, res, vec, eps = 1e-6):
        '''
        Find set of nearest neighbors to vec[0:dim-1]. Their indices are stored
        inside the result object.
        
        Arguments:
            res = the result object in which the indices of the
                  nearest-neighbors are stored
                  
            vec = the test point for which to search the nearest neighbors
         
        Returns true if neighbours where found
        '''
        if self.npoints == 0:
            raise ValueError('no points')
            
        if self.root is None:
            raise ValueError('index not created')
        
        epsError = 1 + eps
        
        dists, distsq = self.computeInitialDistances(vec)
        
        self.searchLevel(res, vec, self.root, distsq, dists, epsError)
        return res.isFull()
    
    def findRange(self, res, minv, maxv):
        '''
        Performs a range search on the range bounded by minv and maxv.
        Their indices are stored inside the result object.
        
        Arguments:
            res = the result object in which the indices of the
                  range search are stored
                  
            minv = lower range value
            maxv = maximum range value
         
        Returns true if points where found
        '''
        if self.npoints == 0:
            raise ValueError('no points')
            
        if self.root is None:
            raise ValueError('index not created')
        
        self.searchRange(res, minv, maxv, self.root)
        return res.isFull()
        
    def searchRange(self, res, minv, maxv, node):
        '''
        Performs range search in tree starting from a node.
        '''
        # If this is a leaf node, then do check and return.
        if node.child1 is None and node.child2 is None:
            for i in range(node.left, node.right):
                index = self.ind[i]
                pnt = self.pnts[index]
                
                inside = True
                for j in range(self.DIM):
                    val = pnt[j]
                    if val < minv[j] or val > maxv[j]:
                        inside = False
                        break
                
                if inside:
                    res.addPoint(index)
            return
            
        idx = node.divfeat
        
        if node.divlow >= minv[idx]:
            self.searchRange(res, minv, maxv, node.child1)
        
        if node.divhigh <= maxv[idx]:
            self.searchRange(res, minv, maxv, node.child2)
        
    def searchLevel(self, res, vec, node, mindistsq, dists, epsError):
        '''
        Performs an exact search in the tree starting from a node.
        '''
        # If this is a leaf node, then do check and return.
        if node.child1 is None and node.child2 is None:
            worst_dist = res.worstDist()
            for i in range(node.left, node.right):
                index = self.ind[i]
                dist = self.distance(vec, index)
                if dist < worst_dist:
                    res.addPoint(dist, index)
            return
        
        # Which child branch should be taken first?
        idx = node.divfeat
        val = vec[idx]
        
        diff1 = val - node.divlow
        diff2 = val - node.divhigh
        
        if diff1 + diff2 < 0:
            bestChild = node.child1
            otherChild = node.child2
            cut_dist = self.accum_dist(val, node.divhigh)
        else:
            bestChild = node.child2
            otherChild = node.child1
            cut_dist = self.accum_dist(val, node.divlow)
        
        # Call recursively to search next level down.
        self.searchLevel(res, vec, bestChild, mindistsq, dists, epsError)
        
        dst = dists[idx]
        mindistsq = mindistsq + cut_dist - dst
        dists[idx] = cut_dist
        
        if mindistsq*epsError <= res.worstDist():
            self.searchLevel(res, vec, otherChild, mindistsq, dists, epsError)
        
        dists[idx] = dst
    
    def accum_dist(self, a, b):
        return (a-b)*(a-b)
    
    def distance(self, p1, idx):
        '''
        Square distance norm
        '''
        d0 = p1[0] - self.pnts[idx, 0]
        d1 = p1[1] - self.pnts[idx, 1]
        d2 = p1[2] - self.pnts[idx, 2]
        return d0*d0+d1*d1+d2*d2
        
if __name__ == '__main__':
    import sys
    import timeit
    
    #'''
    pnts = np.array((
        (1.,2.,3.),
        (4.,5.,6.),
        (7.,8.,9.)
    ))
    
    tree = KDTree(pnts, maxLeafSize = 1)
    tree.build()
    print(tree)
    
    '''
    res = RadiusResultSet(radius = 1.)
    tree.findNeighbors(res, np.array((4.,5.,6.)))
    print(res.res)
    
    res = KNNResultSet(3)
    tree.findNeighbors(res, np.array((1.,2.,3.5)))
    print(res)
    for data in res:
        print(data)
    '''
    
    res = RangeResultSet()
    tree.findRange(res, np.array((0.,0.,0.)), np.array((5.,5.,6.)))
    print(res)
    for data in res:
        print(data)
    
    res = RangeResultSet()
    tree.findRange(res, np.array((10.,10.,10.)), np.array((50.,50.,60.)))
    print(res)
    for data in res:
        print(data)
        
    #'''
    ''''
    sys.setrecursionlimit(100000000)
    
    N = 100000
    pnts = 10.*np.random.rand(N,3)
    
    tree = KDTree(pnts, maxLeafSize = 25)
    
    print(timeit.timeit(lambda: tree.build(), number=1))
    
    res = KNNResultSet(1)
    tree.findNeighbors(res, np.array((5.,5.,5.)))
    print(res)
    for data in res:
        print(data)
    '''
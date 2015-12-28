# -*- coding: utf-8 -*-
#cython: embedsignature=True
#
# Kd-Tree
#
from libc.math cimport HUGE_VAL
from libc.stdint cimport uint32_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libcpp.vector cimport vector

cimport cython
from cython.view cimport array as cvarray

from cpython cimport array
import array

cdef class Adapter:
    '''
    Base class for accessing points
    '''
    property shape:
        def __get__(self):
            raise NotImplemented
            
    cdef double get(self, uint row, int col):
        raise NotImplemented

cdef class FAdapter(Adapter):
    '''
    Adapter for accessing float array
    '''
    def __init__(self, points):
        self.points = points
        
    property shape:
        def __get__(self):
            return self.points.shape
            
    cdef double get(self, uint row, int col):
        return <double>self.points[row, col]
        

cdef class DAdapter(Adapter):
    '''
    Adapter for accessing float array
    '''
    def __init__(self, points):
        self.points = points
        
    property shape:
        def __get__(self):
            return self.points.shape
            
    cdef double get(self, uint row, int col):
        return self.points[row, col]
    
cdef class ResultSet:
    '''
    Result set abstract class
    '''
    cpdef bint isFull(self):
        raise NotImplemented
    
    cpdef addPoint(self,  uint index, double dist):
        raise NotImplemented
    
    cpdef double worstDist(self):
        raise NotImplemented
        
cdef class KNNResultSet(ResultSet):
    '''
    Result set for N neighbour search
    '''
    def __init__(self, uint capacity = 1):
        cdef uint i
        
        if capacity == 0:
            raise ValueError('capacity = 0')
            
        self.capacity = capacity
        self.count = 0
        
        self._ind.reserve(capacity)
        self._dists.reserve(capacity)
        
        # initialize memory
        for i in range(capacity):
            self._ind[i] = 0
            self._dists[i] = 0.
            
        # ensure we have first match
        self._dists[capacity-1] = HUGE_VAL
    
    property ind:
        def __get__(self):
            if self.count == 0:
                return array.array('I',())
            
            cdef cvarray ret = cvarray(
                shape = (self.count,),
                itemsize = sizeof(uint),
                format = "I",
                allocate_buffer=False
            )
            ret.data = <char *> &self._ind[0]
            return ret
            
    
    property dists:
        def __get__(self):
            if self.count == 0:
                return array.array('d',())
                
            cdef cvarray ret = cvarray(
                shape = (self.count,),
                itemsize = sizeof(double),
                format = "d",
                allocate_buffer=False
            )
            ret.data = <char *> &self._dists[0]
            return ret
            
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
        cdef uint i
        for i in range(self.count):
            yield self._ind[i], self._dists[i]
    
    def __reversed__(self):
        '''
        Reversed iteration over values in Bitset
        '''
        cdef uint i
        for i in reversed(range(self.count)):
            yield self._ind[i], self._dists[i]
                
    cpdef bint isFull(self):
        return self.count == self.capacity
    
    cpdef addPoint(self, uint index, double dist):
        cdef uint i
        
        i = self.count
        while i > 0:
            if self._dists[i-1] > dist:
                if i < self.capacity:
                    self._dists[i] = self._dists[i-1]
                    self._ind[i] = self._ind[i-1]
            else:
                break
            i -= 1
        
        if i < self.capacity:
            self._dists[i] = dist
            self._ind[i] = index
        
        if self.count < self.capacity:
            self.count += 1
    
    cpdef double worstDist(self):
        return self._dists[self.capacity - 1]
            
cdef class RadiusResultSet(ResultSet):
    '''
    Result set for performing radius based search.
    '''
    def __init__(self, double radius):
        if radius <= 0:
            raise ValueError('radius <= 0')
        self.radius = radius
        self.radiusSQ = radius*radius

    property ind:
        def __get__(self):
            if self.size() == 0:
                return array.array('I',())
                
            cdef cvarray ret = cvarray(
                shape = (self.size(),),
                itemsize = sizeof(uint),
                format = "I",
                allocate_buffer=False
            )
            ret.data = <char *> &self._ind[0]
            return ret
            
    
    property dists:
        def __get__(self):
            if self.size() == 0:
                return array.array('d',())
                
            cdef cvarray ret = cvarray(
                shape = (self.size(),),
                itemsize = sizeof(double),
                format = "d",
                allocate_buffer=False
            )
            ret.data = <char *> &self._dists[0]
            return ret
            
    def __repr__(self):
        args = self.size(), self.radius
        return 'RadiusResultSet(size=%d, radius=%g)' % args
    
    __str__ = __repr__
    
    def __len__(self):
        '''
        Return length of result set
        '''
        return self.size()
    
    def __bool__(self):
        '''
        Test if result set contains any data
        '''
        return bool(self.size())
    
    __hash__ = None
    
    def __iter__(self):
        '''
        Iterate over result
        '''
        cdef uint i
        for i in range(self.size()):
            yield self._ind[i], self._dists[i]
    
    def __reversed__(self):
        '''
        Reversed iteration over values
        '''
        cdef uint i
        for i in reversed(range(self.size())):
            yield self._ind[i], self._dists[i]
            
    cpdef clear(self):
        self._ind.clear()
        self._dists.clear()
    
    cpdef uint size(self):
        return self._ind.size()
    
    cpdef bint isFull(self):
        return True
    
    cpdef addPoint(self, uint index, double dist):
        if dist < self.radiusSQ:
            self._ind.push_back(index)
            self._dists.push_back(dist)
    
    cpdef double worstDist(self):
        return self.radiusSQ

cdef class RangeResultSet(ResultSet):
    '''
    Result set for performing range based search.
    '''
    def __init__(self):
        pass
    
    property ind:
        def __get__(self):
            if self.size() == 0:
                return array.array('I',())
                
            cdef cvarray ret = cvarray(
                shape = (self.size(),),
                itemsize = sizeof(uint),
                format = "I",
                allocate_buffer=False
            )
            ret.data = <char *> &self._ind[0]
            return ret
            
    def __repr__(self):
        return 'RangeResultSet(size=%d)' % self.size()
    
    __str__ = __repr__
    
    def __len__(self):
        '''
        Return length of result set
        '''
        return self.size()
    
    def __bool__(self):
        '''
        Test if result set contains any data
        '''
        return bool(self.size())
    
    __hash__ = None
    
    def __iter__(self):
        '''
        Iterate over result
        '''
        cdef uint i
        for i in range(self.size()):
            yield self._ind[i]
    
    def __reversed__(self):
        '''
        Reversed iteration over values in Bitset
        '''
        cdef uint i
        for i in reversed(range(self.size())):
            yield self._ind[i]
            
    cpdef clear(self):
        self._ind.clear()
    
    cpdef uint size(self):
        return self._ind.size()
    
    cpdef addPoint(self, uint index, double dist):
        self._ind.push_back(index)
        
cdef class KDTree:
    '''
    KD search tree optimized for point clouds of 3d points.
    '''
    def __init__(self, points, bbox = None, int maxLeafSize = 10):
        cdef uint i
        
        if isinstance(points, Adapter):
            self.points = points
        else:
            try:
                self.points = DAdapter(points)
            except ValueError:
                self.points = FAdapter(points)
            
        self.npoints = self.points.shape[0]
        self.dim = self.points.shape[1]
        
        if bbox is None:
            self.bbox = self.findBBox()
        else:
            self.bbox = bbox
            if self.bbox.shape[0] != 2 or self.bbox.shape[1] != self.dim:
                raise ValueError('expected bbox of shape (2, dim)')
        
        if maxLeafSize < 1:
            raise ValueError('maxLeafSize < 1')
        self.maxLeafSize = maxLeafSize
        
        # set up indices array
        self.ind = cvarray(shape=(self.npoints,), itemsize=sizeof(uint), format="I")
        for i in range(self.npoints):
            self.ind[i] = i
            
    def __dealloc__(self):
        self.free(self.root)
    
    cdef free(self, Node *node):
        '''
        Recurse tree to free memory
        '''
        if node == NULL:
            return
        self.free(node.child1)
        self.free(node.child2)
        PyMem_Free(node)
        
    def __repr__(self):
        args = self.npoints, self.dim, not self.root == NULL, \
               self.maxLeafSize
        msg = 'KDTree(npoints=%d, dim=%d, hasIndex=%s, maxLeafSize=%d)'
        return msg % args
    
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
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double[:,:] findBBox(self):
        '''
        Calculate bounding box from points
        '''
        cdef double[:,:] bbox
        cdef uint i, j
        
        bbox = cvarray(shape=(2, self.dim), itemsize=sizeof(double), format="d")
        
        for i in range(self.dim):
            bbox[0, i] = self.points.get(0, i)
            bbox[1, i] = self.points.get(0, i)
            
        for i in range(1, self.npoints):
            for j in range(self.dim):
                bbox[0, j] = min(bbox[0, j], self.points.get(i, j))
                bbox[1, j] = max(bbox[1, j], self.points.get(i, j))
        
        return bbox
    
    cpdef build(self):
        '''
        Recursive build the tree structure
        '''
        self.free(self.root)
        self.root = self.divideTree(0, self.npoints, self.bbox)
    
    cdef Node *divideTree(self, uint left, uint right, double[:,:] bbox):
        cdef Node *node
        cdef double[:,:] left_bbox, right_bbox
        cdef double cutval
        cdef uint i, j, idx, cutfeat
        
        # create node
        node = <Node *>PyMem_Malloc(sizeof(Node))
        if node == NULL:
            raise MemoryError()
        
        # If too few exemplars remain, then make this a leaf node.
        if (right - left) <= self.maxLeafSize:
            # mark as leaf node
            node.child1 = node.child2 = NULL
            node.head.lr.left = left
            node.head.lr.right = right
            
            # compute bounding-box of leaf points
            for i in range(self.dim):
                bbox[0,i] = self.points.get(self.ind[left], i)
                bbox[1,i] = self.points.get(self.ind[left], i)
            
            for i in range(left + 1, right):
                for j in range(self.dim):
                    bbox[0,j] = min(bbox[0,j], self.points.get(self.ind[i], j))
                    bbox[1,j] = max(bbox[1,j], self.points.get(self.ind[i], j))
        else:
            self.middleSplit(left, right - left, bbox, &idx, &cutfeat, &cutval)
            
            node.head.sub.divfeat = cutfeat
            
            left_bbox = bbox.copy()
            left_bbox[1,cutfeat] = cutval
            node.child1 = self.divideTree(left, left + idx, left_bbox)
            
            right_bbox = bbox.copy()
            right_bbox[0,cutfeat] = cutval
            node.child2 = self.divideTree(left+idx, right, right_bbox)
            
            node.head.sub.divlow = left_bbox[1, cutfeat]
            node.head.sub.divhigh = right_bbox[0, cutfeat]
            
            for i in range(self.dim):
                bbox[0, i] = min(left_bbox[0, i], right_bbox[0, i])
                bbox[1, i] = max(left_bbox[1, i], right_bbox[1, i])
            
        return node
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef computeMinMax(self, uint ind, uint count, uint element,
                        double *min_element, double *max_element):
        cdef double val
        cdef uint i
        
        min_element[0] = self.points.get(self.ind[ind], element)
        max_element[0] = self.points.get(self.ind[ind], element)
        
        for i in range(1, count):
            val = self.points.get(self.ind[ind + i], element)
            if val < min_element[0]:
                min_element[0] = val
            if val > max_element[0]:
                max_element[0] = val
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef middleSplit(self, uint ind, uint count, double[:,:] bbox,
                       uint *index, uint *cutfeat, double *cutval):
        cdef double split_val, span, max_span, EPS = .00001
        cdef double min_element, max_element, spread, max_spread
        cdef uint i, lim1, lim2
        
        max_span = bbox[1,0] - bbox[0,0]
        for i in range(1, self.dim):
            span = bbox[1,i] - bbox[0,i]
            if span > max_span:
                max_span = span
        
        max_spread = -1
        cutfeat[0] = 0
        for i in range(self.dim):
            span = bbox[1,i] - bbox[0,i]
            if span > (1. - EPS)*max_span:
                self.computeMinMax(ind, count, cutfeat[0], &min_element, &max_element)
                spread = max_element - min_element
                if spread > max_spread:
                    cutfeat[0] = i
                    max_spread = spread
        
        # split in the middle
        split_val = .5*(bbox[1,cutfeat[0]] + bbox[0,cutfeat[0]])
        self.computeMinMax(ind, count, cutfeat[0], &min_element, &max_element)
        
        if split_val < min_element:
            cutval[0] = min_element
        elif split_val > max_element:
            cutval[0] = max_element
        else:
            cutval[0] = split_val
        
        self.planeSplit(ind, count, cutfeat[0], cutval[0], &lim1, &lim2)
        
        if lim1 > count / 2:
            index[0] = lim1
        elif lim2 < count / 2:
            index[0] = lim2
        else:
            index[0] = count / 2
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef planeSplit(self, uint ind, uint count, uint cutfeat, double cutval,
                     uint *lim1, uint *lim2):
        '''
        Subdivide the list of points by a plane perpendicular on axe corresponding
        to the 'cutfeat' dimension at 'cutval' position.
        '''
        cdef uint left, right
        
        left = 0
        right = count - 1
        
        # Move vector indices for left subtree to front of list.
        while True:
            while left <= right and self.points.get(self.ind[ind+left], cutfeat) < cutval:
                left += 1
            
            while right and left <= right and \
                   self.points.get(self.ind[ind+right], cutfeat) >= cutval:
                right -= 1
            
            if left > right or not right:
                break
            
            self.ind[ind+left], self.ind[ind+right] = self.ind[ind+right], self.ind[ind+left]
            left += 1
            right -= 1
        
        # If either list is empty, it means that all remaining features
        # are identical. Split in the middle to maintain a balanced tree.
        lim1[0] = left
        right = count - 1
        while True:
            while left <= right and self.points.get(self.ind[ind+left], cutfeat) <= cutval:
                left += 1
            
            while right and left <= right and \
                   self.points.get(self.ind[ind+right], cutfeat) > cutval:
                right -= 1
            
            if left > right or not right:
                break
            
            self.ind[ind+left], self.ind[ind+right] = self.ind[ind+right], self.ind[ind+left]
            left += 1
            right -= 1
        
        lim2[0] = left
    
    cdef double accum_dist(self, double a, double b):
        return (a-b)*(a-b)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef double distance(self, double [:] p1, uint idx):
        '''
        Square distance norm
        '''
        cdef double d0 = p1[0] - self.points.get(idx, 0)
        cdef double d1 = p1[1] - self.points.get(idx, 1)
        cdef double d2 = p1[2] - self.points.get(idx, 2)
        return d0*d0+d1*d1+d2*d2
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef computeInitialDistances(self, double[:] vec, double[:] dists,
                                    double *distsq):
        cdef uint i
        
        distsq[0] = 0
        for i in range(self.dim):
            if vec[i] < self.bbox[0, i]:
                dists[i] = self.accum_dist(vec[i], self.bbox[0, i])
                distsq[0] += dists[i]
            if vec[i] > self.bbox[1, i]:
                dists[i] = self.accum_dist(vec[i], self.bbox[1, i])
                distsq[0] += dists[i]
    
    cpdef closest(self, pnt):
        '''
        Find index and distance to closest point.
        
        raise ValueError if tree is empty
        '''
        res = KNNResultSet(1)
        self.findNeighbors(res, pnt)
        return res.ind[0], res.dists[0]
    
    cpdef RadiusResultSet search(self, pnt, radius, double eps = 1e-6):
        '''
        Find index and distance to points inside sphere
        defined by point and radius.
        
        raise ValueError if tree is empty
        '''
        res = RadiusResultSet(radius)
        self.findNeighbors(res, pnt, eps)
        return res
    
    cpdef RangeResultSet searchBBox(self, minv, maxv):
        '''
        Find index and distance to points inside bounding
        box given by minv and maxv.
        
        raise ValueError if tree is empty
        '''
        res = RangeResultSet()
        self.findRange(res, minv, maxv)
        return res
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bint findNeighbors(self, ResultSet res, pnt, double eps = 1e-6):
        '''
        Find set of nearest neighbors to vec[0:dim-1]. Their indices are stored
        inside the result object.
        
        Arguments:
            res = the result object in which the indices of the
                  nearest-neighbors are stored
                  
            pnt = the test point for which to search the nearest neighbors
         
        Returns true if neighbours where found
        '''
        cdef double[:] vec
        cdef double[:] dists
        cdef double distsq, epsError
        cdef uint i
        
        if self.npoints == 0:
            raise ValueError('no points')
            
        if self.root == NULL:
            raise ValueError('index not created')
        
        try:
            vec = pnt
        except TypeError:
            vec = cvarray(shape=(self.dim,), itemsize=sizeof(double), format="d")
            for i in range(self.dim):
                vec[i] = pnt[i]
        
        epsError = 1 + eps
        
        dists = cvarray(shape=(self.dim,), itemsize=sizeof(double), format="d")
        self.computeInitialDistances(vec, dists, &distsq)
        
        self.searchLevel(res, vec, self.root, distsq, dists, epsError)
        return res.isFull()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bint findRange(self, ResultSet res, minv, maxv):
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
        cdef double[:] _minv
        cdef double[:] _maxv
        cdef uint i
        
        if self.npoints == 0:
            raise ValueError('no points')
            
        if self.root == NULL:
            raise ValueError('index not created')
                
        try:
            _minv = minv
        except TypeError:
            _minv = cvarray(shape=(self.dim,), itemsize=sizeof(double), format="d")
            for i in range(self.dim):
                _minv[i] = minv[i]
        
        try:
            _maxv = maxv
        except TypeError:
            _maxv = cvarray(shape=(self.dim,), itemsize=sizeof(double), format="d")
            for i in range(self.dim):
                _maxv[i] = maxv[i]
        
        # check if minv and maxv is valid
        for i in range(self.dim):
            if _minv[i] > _maxv[i]:
                raise ValueError('minv > maxv')
            
        self.searchRangeLevel(res, _minv, _maxv, self.root)
        return res.size()
            
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef searchLevel(self, ResultSet res, double[:] vec, Node *node,
                       double mindistsq, double[:] dists, double epsError):
        '''
        Performs an exact search in the tree starting from a node.
        '''
        cdef Node *bestChild
        cdef Node *otherChild
        cdef double dist, worst_dist, cut_dist
        cdef double val, diff1, diff2, dst
        cdef uint i, idx, index
        
        # If this is a leaf node, then do check and return.
        if node.child1 == NULL and node.child2 == NULL:
            worst_dist = res.worstDist()
            for i in range(node.head.lr.left, node.head.lr.right):
                index = self.ind[i]
                dist = self.distance(vec, index)
                if dist < worst_dist:
                    res.addPoint(index, dist)
            return
        
        # Which child branch should be taken first?
        idx = node.head.sub.divfeat
        val = vec[idx]
        
        diff1 = val - node.head.sub.divlow
        diff2 = val - node.head.sub.divhigh
        
        if diff1 + diff2 < 0:
            bestChild = node.child1
            otherChild = node.child2
            cut_dist = self.accum_dist(val, node.head.sub.divhigh)
        else:
            bestChild = node.child2
            otherChild = node.child1
            cut_dist = self.accum_dist(val, node.head.sub.divlow)
        
        # Call recursively to search next level down.
        self.searchLevel(res, vec, bestChild, mindistsq, dists, epsError)
        
        dst = dists[idx]
        mindistsq = mindistsq + cut_dist - dst
        dists[idx] = cut_dist
        
        if mindistsq*epsError <= res.worstDist():
            self.searchLevel(res, vec, otherChild, mindistsq, dists, epsError)
        
        dists[idx] = dst
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef searchRangeLevel(self, ResultSet res, double[:] minv, double[:] maxv,
                          Node *node):
        '''
        Performs range search in tree starting from a node.
        '''
        cdef double val
        cdef uint i, idx, index
        cdef bint inside
        
        # If this is a leaf node, then do check and return.
        if node.child1 == NULL and node.child2 == NULL:
            for i in range(node.head.lr.left, node.head.lr.right):
                index = self.ind[i]
                inside = True
                for j in range(self.dim):
                    val = self.points.get(index, j)
                    if val < minv[j] or val > maxv[j]:
                        inside = False
                        break
                
                if inside:
                    res.addPoint(index, 0.)
            return
        
        idx = node.head.sub.divfeat
        
        if node.head.sub.divlow >= minv[idx]:
            self.searchRangeLevel(res, minv, maxv, node.child1)
        
        if node.head.sub.divhigh <= maxv[idx]:
            self.searchRangeLevel(res, minv, maxv, node.child2)
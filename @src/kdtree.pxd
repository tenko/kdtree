# -*- coding: utf-8 -*-
#cython: embedsignature=True
#
# Kd-Tree
#
from libc.stdint cimport uint32_t
from libcpp.vector cimport vector

cimport cython
import cython

from cython.view cimport array as cvarray

from cpython cimport array

ctypedef unsigned int uint

# Node structure definition
cdef struct _Leaf:
    uint32_t left
    uint32_t right

ctypedef _Leaf Leaf

cdef struct _SubDiv:
    int divfeat
    double divlow
    double divhigh

ctypedef _SubDiv SubDiv

cdef union _NodeHead:
    Leaf lr
    SubDiv sub

ctypedef _NodeHead NodeHead

cdef struct _Node:
    NodeHead head
    _Node *child1
    _Node *child2

ctypedef _Node Node

cdef class Adapter:
    '''
    Base class for accessing points
    '''
    cdef double get(self, uint row, int col)

cdef class FAdapter(Adapter):
    '''
    Adapter for accessing float array
    '''
    cdef readonly float[:,:] points

cdef class DAdapter(Adapter):
    '''
    Adapter for accessing float array
    '''
    cdef readonly double[:,:] points
    
cdef class ResultSet:
    '''
    Result set abstract class
    '''
    cpdef bint isFull(self)
    
    cpdef addPoint(self, uint index, double dist)
    
    cpdef double worstDist(self)

cdef class KNNResultSet(ResultSet):
    '''
    Result set for N neighbour search
    '''
    cdef vector[uint] _ind
    cdef vector[double] _dists
    cdef readonly uint capacity
    cdef readonly uint count

cdef class RadiusResultSet(ResultSet):
    '''
    Result set for performing radius based search.
    '''
    cdef readonly double radius
    cdef double radiusSQ
    cdef vector[uint] _ind
    cdef vector[double] _dists
    
    cpdef clear(self)
    cpdef uint size(self)

cdef class RangeResultSet(ResultSet):
    '''
    Result set for performing range based search.
    '''
    cdef vector[uint] _ind
    
    cpdef clear(self)
    cpdef uint size(self)
    
cdef class KDTree:
    '''
    KD search tree optimized for point clouds of 3d points.
    '''
    cdef Node *root
    
    cdef readonly Adapter points
    cdef readonly double[:,:] bbox
    cdef readonly uint[:] ind
    
    cdef readonly uint npoints
    cdef readonly int dim
    cdef readonly int maxLeafSize
    
    cdef free(self, Node *node)
    
    cdef double[:,:] findBBox(self)
    
    cpdef build(self)
    
    cdef Node *divideTree(self, uint left, uint right, double[:,:] bbox)
    
    cdef computeMinMax(self, uint ind, uint count, uint element,
                        double *min_element, double *max_element)
    
    cdef middleSplit(self, uint ind, uint count, double[:,:] bbox,
                       uint *index, uint *cutfeat, double *cutval)
    
    cdef planeSplit(self, uint ind, uint count, uint cutfeat, double cutval,
                     uint *lim1, uint *lim2)
    
    cdef double accum_dist(self, double a, double b)
    
    cdef double distance(self, double [:] p1, uint idx)
    
    cdef computeInitialDistances(self, double[:] vec, double[:] dists,
                                 double *distsq)
    
    cpdef closest(self, pnt)
    cpdef RadiusResultSet search(self, pnt, radius, double eps = ?) 
    cpdef RangeResultSet searchBBox(self, minv, maxv)
    
    cpdef bint findNeighbors(self, ResultSet res, pnt, double eps = ?)
    cpdef bint findRange(self, ResultSet res, minv, maxv)
    
    cdef searchRangeLevel(self, ResultSet res, double[:] minv, double[:] maxv,
                          Node *node)
    
    cdef searchLevel(self, ResultSet res, double[:] vec, Node *node,
                       double mindistsq, double[:] dists, double epsError)
        
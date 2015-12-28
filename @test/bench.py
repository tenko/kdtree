# -*- coding: utf-8 -*-
from random import randint, seed
from timeit import Timer

import numpy as np

def asciitable(rows):
    # From : https://gist.github.com/lonetwin/4721748
    # - figure out column widths
    widths = [len(max(columns, key=len)) for columns in zip(*rows)]
    
    def separator():
        print('-+-'.join( '-' * width for width in widths ))
    
    separator()
    
    # - print the header
    header, data = rows[0], rows[1:]
    print(
        ' | '.join(format(title, "%ds" % width) for width, title in zip(widths, header))
    )
    
    separator()

    # - print the data
    for row in data:
        print(
            " | ".join(format(cdata, "%ds" % width) for width, cdata in zip(widths, row))
        )
    
    separator()

DATA = None
if __name__ == '__main__':
    seed(42)
    
    HEADING = ('Test', 'cKDTree', 'KDTree', 'Ratio')
    rows = [HEADING]
    
    SETUP_CKDTREE = """
from __main__ import DATA
from scipy.spatial import cKDTree
kdtree1 = cKDTree(DATA, leafsize=10)
import numpy as np
pnt = np.array((.5,.5,.5))
    """
    
    SETUP_KDTREE = """
from __main__ import DATA
import numpy as np
from kdtree import KDTree, KNNResultSet
pnt = np.array((.5,.5,.5))
kdtree2 = KDTree(DATA, maxLeafSize = 10)
kdtree2.build()
res2 = KNNResultSet(10)
    """
    M = 10
    
    def run(name, ckdtree_stmt, kdtree_stmt):
        a = Timer(ckdtree_stmt, setup = SETUP_CKDTREE).timeit(number = M)
        b = Timer(kdtree_stmt, setup = SETUP_KDTREE).timeit(number = M)
        ratio = a / b
        rows.append((name, "%g" % a, "%g" % b, "%.1f" % ratio))
        
    for N in (1000, 10000, 100000):
        DATA = pnts = np.random.rand(N,3)
        
        run('Initialize', 'cKDTree(DATA, leafsize=10)', 'tree=KDTree(DATA, maxLeafSize = 10); tree.build()')
        run('Nearest', 'kdtree1.query(pnt, k=10)', 'kdtree2.findNeighbors(res2, pnt)')
        
        print(" DATA SIZE %d" % N)
        asciitable(rows)
        print("\n")
        
        rows.clear()
        rows.append(HEADING)
    
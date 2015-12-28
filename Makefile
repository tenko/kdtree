PYVERSION=3.5
PYPREFIX=/mingw64
INCLUDES=-I@src -I$(PYPREFIX)/include/python$(PYVERSION)m
LIBS=-L@build -L$(PYPREFIX)/lib/python$(PYVERSION)

all: @build/kdtree-cpython-35m.dll

clean:
	@echo Cleaning
	@rm -f @build/*

test:
	PYTHONPATH=@build python3 @test/test_kdtree.py

bench:
	PYTHONPATH=@build python3 @test/bench.py

@build/kdtree.cpp: @src/*
	cython --cplus @src/kdtree.pyx -o @build/kdtree.cpp

@build/kdtree.o:	@build/kdtree.cpp
	#gcc -c -fPIC $(INCLUDES) $<
	c++ -c -O3 -DNDEBUG -march=native $(INCLUDES) @build/kdtree.cpp -o@build/kdtree.o

@build/kdtree-cpython-35m.dll:	@build/kdtree.o
	c++ -shared @build/kdtree.o $(LIBS) -lpython$(PYVERSION)m -lm -o @build/kdtree-cpython-35m.dll

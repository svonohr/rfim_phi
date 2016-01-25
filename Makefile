HPP_FILES := $(wildcard *.hpp)

CPP_FLAGS := -std=c++11 -qopenmp -O3 -Wall
LD_FLAGS := -lrt -lz -lpthread

.PHONY: all clean
all: ising

ising: ising.cpp SystemManager.cpp
	icc $^ -o $@ $(CPP_FLAGS) $(LD_FLAGS) -qopt-prefetch-distance=6,1

clean:
	rm -f ising


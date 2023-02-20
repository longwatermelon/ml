CXX=g++
CXXFLAGS=-std=c++17 -ggdb -Wall

AR=ar
ARFLAGS=rc

all: src/common.cpp
	$(CXX) $(CXXFLAGS) -c src/common.cpp
	$(AR) $(ARFLAGS) ../lib/libcommon.a common.o

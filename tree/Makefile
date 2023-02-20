CXX=g++
CXXFLAGS=-std=c++17 -ggdb -Wall
INC=-I../common/src/
LDFLAGS=-L../lib/ -lcommon -ldtree

AR=ar
ARFLAGS=rc

lib:
	$(CXX) $(CXXFLAGS) $(INC) -c src/dtree.cpp
	$(AR) $(ARFLAGS) ../lib/libdtree.a dtree.o

%: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) $< $(LDFLAGS)

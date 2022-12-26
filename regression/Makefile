CXX=g++
CXXFLAGS=-std=c++17 -ggdb -Wall
INC=-I../graph/src/
LDFLAGS=-L../lib -lSDL2 -lgraph

.PHONY: graph

all: poly

%: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $^ src/common.cpp
	$(CXX) $(CXXFLAGS) $(INC) $@.o common.o $(LDFLAGS)

graph:
	$(MAKE) -C ../graph

clean:
	-rm *.o a.out
	$(MAKE) clean -C ../graph


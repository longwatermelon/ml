CXX=g++
CXXFLAGS=-std=c++17 -ggdb -Wall
INC=-I../graph/src/
LDFLAGS=-L../lib -lSDL2 -lgraph

.PHONY: graph

all: logistic

%: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $^ src/impl.cpp
	$(CXX) $(CXXFLAGS) $@.o impl.o $(LDFLAGS)

graph:
	$(MAKE) -C ../graph

clean:
	-rm *.o a.out
	$(MAKE) clean -C ../graph


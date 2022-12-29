CXX=g++
CXXFLAGS=-std=c++17 -ggdb -Wall
INC=-I../graph/src/
LDFLAGS=-L../lib -lSDL2 -lgraph

AR=ar
ARFLAGS=rc

.PHONY: graph

all: multilinear

%: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $^ src/impl.cpp
	$(CXX) $@.o impl.o $(LDFLAGS)

lib:
	$(CXX) $(CXXFLAGS) -c src/impl.cpp
	$(AR) $(ARFLAGS) ../lib/libreg.a impl.o

graph:
	$(MAKE) -C ../graph

clean:
	-rm *.o a.out
	$(MAKE) clean -C ../graph


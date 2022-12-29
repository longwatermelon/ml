CXX=g++
CXXFLAGS=-std=c++17 -ggdb -Wall
INC=-I../graph/src/
LDFLAGS=-L../lib -lSDL2 -lgraph

AR=ar
ARFLAGS=rc

.PHONY: graph

all: multilogistic

%: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $^ src/impl.cpp src/common.cpp
	$(CXX) $@.o impl.o common.o $(LDFLAGS)

lib:
	$(CXX) $(CXXFLAGS) -c src/impl.cpp
	$(AR) $(ARFLAGS) ../lib/libreg.a impl.o

graph:
	$(MAKE) -C ../graph

clean:
	-rm *.o a.out
	$(MAKE) clean -C ../graph


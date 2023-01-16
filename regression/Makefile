CXX=g++
CXXFLAGS=-std=c++17 -ggdb -Wall
INC=-I../graph/src/
LDFLAGS=-L../lib -lSDL2 -lgraph

AR=ar
ARFLAGS=rc

.PHONY: graph

all: softmax

%: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $^ src/reg.cpp src/common.cpp
	$(CXX) $@.o impl.o common.o $(LDFLAGS)

lib:
	$(CXX) $(CXXFLAGS) -c src/reg.cpp
	$(AR) $(ARFLAGS) ../lib/libreg.a reg.o

graph:
	$(MAKE) -C ../graph

clean:
	-rm *.o a.out
	$(MAKE) clean -C ../graph


CXX=g++
CXXFLAGS=-std=c++17 -ggdb -Wall
INC=-I../graph/src/ -I../common/src/
LDFLAGS=-L../lib -lSDL2 -lgraph -lreg

AR=ar
ARFLAGS=rc

.PHONY: graph

all: softmax

%: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $^
	$(CXX) $@.o $(LDFLAGS)

lib:
	$(CXX) $(CXXFLAGS) $(INC) -c src/reg.cpp
	$(AR) $(ARFLAGS) ../lib/libreg.a reg.o

graph:
	$(MAKE) -C ../graph

clean:
	-rm *.o a.out
	$(MAKE) clean -C ../graph


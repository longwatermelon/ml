CXX=g++
CXXFLAGS=-std=c++17 -ggdb -Wall
INC=-I../regression/src -I../common/src -I/usr/include/eigen3
LDFLAGS=-L../lib -lreg

SRC=$(wildcard src/*.cpp)
OBJS=$(addprefix obj/, $(SRC:.cpp=.o))

.PHONY: graph

all: cats

%: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $^ src/deepnn.cpp
	$(CXX) $@.o deepnn.o $(LDFLAGS)

lib:
	$(CXX) $(CXXFLAGS) -c src/deepnn.cpp
	$(AR) $(ARFLAGS) ../lib/libnn.a deepnn.o

reg:
	$(MAKE) -C ../regression lib

clean:
	-rm *.o a.out


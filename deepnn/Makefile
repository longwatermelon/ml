CXX=g++
CXXFLAGS=-std=c++17 -ggdb -Wall
INC=-I../common/src -I../graph/src/
LDFLAGS=-L../lib -lgraph -lSDL2

SRC=$(wildcard src/*.cpp)
OBJS=$(addprefix obj/, $(SRC:.cpp=.o))

.PHONY: graph

all: cats

%: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $^
	$(CXX) $@.o $(LDFLAGS) -lnn

lib:
	$(CXX) $(CXXFLAGS) $(INC) -c src/deepnn.cpp
	$(AR) $(ARFLAGS) ../lib/libnn.a deepnn.o

reg:
	$(MAKE) -C ../regression lib

clean:
	-rm *.o a.out


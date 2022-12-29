CXX=g++
CXXFLAGS=-std=c++17 -ggdb -Wall
INC=
LDFLAGS=-L../lib -lSDL2 -lreg

.PHONY: reg

all: cat-recognition

%: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $^ src/impl.cpp
	$(CXX) $@.o impl.o $(LDFLAGS)

reg:
	$(MAKE) lib -C ../regression/


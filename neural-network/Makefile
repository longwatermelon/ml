CXX=g++
CXXFLAGS=-std=c++17 -ggdb -Wall
INC=
LDFLAGS=-lSDL2

.PHONY: reg

%: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) -c $^ src/impl.cpp
	$(CXX) $@.o impl.o $(LDFLAGS)

reg:
	$(MAKE) lib -C ../regression/


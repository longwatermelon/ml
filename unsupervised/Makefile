CXX=g++
CXXFLAGS=-std=c++17 -ggdb -Wall
INC=-I../common/src/ -I../graph/src/
LDFLAGS=-L../lib/ -lcommon -lgraph -lSDL2

%: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) $< $(LDFLAGS)

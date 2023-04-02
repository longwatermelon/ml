CXX=g++
CXXFLAGS=-std=c++17 -ggdb -Wall
INC=-I../common/src/ -I../graph/src/ -I../regression/src/
LDFLAGS=-L../lib/ -lcommon -lgraph -lSDL2 -lm -lunsupervised -lreg

AR=ar
ARFLAGS=rc

%: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INC) $< $(LDFLAGS)

lib:
	$(CXX) $(CXXFLAGS) $(INC) -c src/unsupervised.cpp
	$(AR) $(ARFLAGS) ../lib/libunsupervised.a unsupervised.o

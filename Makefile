.PHONY: all

CXX ?= g++
CXXFLAGS ?= -std=c++17 -ggdb -Wall -O2
EMEL_SOURCES := src/emel/*.cpp

all: cnn-mnist

%: src/%.cpp
	$(CXX) $< $(EMEL_SOURCES) $(CXXFLAGS)

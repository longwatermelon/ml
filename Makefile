.PHONY: all

CXX ?= g++
CXXFLAGS ?= -std=c++17 -ggdb -Wall -O2
EMEL_SOURCES := src/emel/*.cpp

all: shakespeare-gpt

%: src/%.cpp
	$(CXX) $< $(EMEL_SOURCES) $(CXXFLAGS)

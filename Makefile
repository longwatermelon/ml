.PHONY: all

all:
	g++ src/fnn-demo.cpp src/emel/*.cpp -std=c++17 -ggdb -Wall -O2

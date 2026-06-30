.PHONY: all

all:
	g++ src/cnn-demo.cpp src/emel/*.cpp -std=c++17 -ggdb -Wall -O2

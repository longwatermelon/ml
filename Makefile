.PHONY: all

all:
	g++ src/fnn-demo.cpp src/lib/*.cpp -std=c++17 -ggdb -Wall

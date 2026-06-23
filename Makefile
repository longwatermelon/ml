.PHONY: all

all:
	g++ src/main.cpp src/lib/*.cpp -std=c++17 -ggdb -Wall

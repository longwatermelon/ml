CXX=g++
CXXFLAGS=-std=c++17 -ggdb -Wall
LDFLAGS=-lSDL2

AR=ar
ARFLAGS=rc

SRC=$(wildcard src/*.cpp)
OBJS=$(addprefix obj/, $(SRC:.cpp=.o))

.PHONY: dirs clean
all: dirs target

target: $(OBJS)
	$(AR) $(ARFLAGS) ../lib/libgraph.a $^

obj/src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

dirs:
	mkdir -p obj/src

clean:
	-rm -rf obj/


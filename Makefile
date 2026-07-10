.PHONY: all test

CXX ?= g++
CXXFLAGS ?= -std=c++17 -ggdb -Wall -O2
DOCTEST_CXXFLAGS ?= $(shell pkg-config --cflags doctest 2>/dev/null)
EMEL_SOURCES := $(sort $(shell find src/emel -type f -name '*.cpp'))
TEST_SOURCES := $(sort $(shell find tests -type f -name '*.cpp'))
SOURCE_DIRS := $(sort $(shell find src/emel tests -type d))
TEST_CPPFLAGS := -Isrc $(DOCTEST_CXXFLAGS)
TEST_BIN := .cache/tests
TEST_OBJECT_DIR := .cache/test-objects
TEST_OBJECTS := $(patsubst %.cpp,$(TEST_OBJECT_DIR)/%.o,$(EMEL_SOURCES) $(TEST_SOURCES))
TEST_DEPENDENCIES := $(TEST_OBJECTS:.o=.d)
TEST_ARGS ?=

%: src/%.cpp
	$(CXX) $< $(EMEL_SOURCES) $(CXXFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN) $(TEST_ARGS)

$(TEST_BIN): $(TEST_OBJECTS) $(SOURCE_DIRS)
	mkdir -p $(dir $@)
	$(CXX) $(TEST_OBJECTS) $(CXXFLAGS) -o $@

$(TEST_OBJECT_DIR)/%.o: %.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(TEST_CPPFLAGS) -MMD -MP -c $< -o $@

-include $(TEST_DEPENDENCIES)

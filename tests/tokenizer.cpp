#include <doctest/doctest.h>
#include "emel/tokenizer.h"
#include <stdexcept>

TEST_SUITE_BEGIN("tokenizer");

TEST_CASE("character tokenizer builds a sorted unique vocabulary") {
    CharTokenizer tokenizer("cabca");

    CHECK(tokenizer.vocab_size() == 3);
    CHECK(tokenizer.encode("abc") == vec<int>{0, 1, 2});
    CHECK(tokenizer.encode("caba") == vec<int>{2, 0, 1, 0});
}

TEST_CASE("character tokenizer round trips text") {
    CharTokenizer tokenizer("hello, world!\n");
    string text = "world! hello\n";

    vec<int> ids = tokenizer.encode(text);

    CHECK(tokenizer.decode(ids) == text);
}

TEST_CASE("character tokenizer handles empty input") {
    CharTokenizer tokenizer("");

    CHECK(tokenizer.vocab_size() == 0);
    CHECK(tokenizer.encode("").empty());
    CHECK(tokenizer.decode({}).empty());
}

TEST_CASE("character tokenizer rejects unknown characters") {
    CharTokenizer tokenizer("abc");

    CHECK_THROWS_AS(tokenizer.encode("abd"), std::runtime_error);
}

TEST_CASE("character tokenizer rejects invalid token ids") {
    CharTokenizer tokenizer("abc");

    CHECK_THROWS_AS(tokenizer.decode({-1}), std::runtime_error);
    CHECK_THROWS_AS(tokenizer.decode({tokenizer.vocab_size()}), std::runtime_error);
}

TEST_CASE("character tokenizer supports every byte value") {
    string corpus;
    vec<int> expected_ids;
    for (int value = 0; value < 256; ++value) {
        corpus.push_back(static_cast<char>(value));
        expected_ids.push_back(value);
    }
    CharTokenizer tokenizer(corpus);

    CHECK(tokenizer.vocab_size() == 256);
    CHECK(tokenizer.encode(corpus) == expected_ids);
    CHECK(tokenizer.decode(expected_ids) == corpus);
}

TEST_SUITE_END;

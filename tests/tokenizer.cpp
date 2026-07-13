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

TEST_CASE("bpe supports every byte value") {
    string corpus;
    vec<int> expected_ids;
    for (int value = 0; value < 256; ++value) {
        corpus.push_back(static_cast<char>(value));
        expected_ids.push_back(value);
    }
    BPETokenizer tokz(corpus, 0);

    CHECK(tokz.vocab_size() == 256);
    CHECK(tokz.encode(corpus) == expected_ids);
    CHECK(tokz.decode(expected_ids) == corpus);
}

TEST_CASE("bpe learns the most frequent pair") {
    BPETokenizer tokenizer("abababab", 1);
    vec<int> ids = tokenizer.encode("abxabab");

    CHECK(tokenizer.vocab_size() == 257);
    CHECK(ids.size() == 4);
    CHECK(tokenizer.decode(ids) == "abxabab");
}

TEST_CASE("bpe applies learned merges in order") {
    BPETokenizer tokenizer("abababab", 2);
    vec<int> ids = tokenizer.encode("abababab");

    CHECK(tokenizer.vocab_size() == 258);
    CHECK(ids.size() == 2);
    CHECK(tokenizer.encode("abab").size() == 1);
    CHECK(tokenizer.decode(ids) == "abababab");
}

TEST_CASE("bpe merges repeated pairs without overlap") {
    BPETokenizer tokenizer("aaaa", 1);
    vec<int> ids = tokenizer.encode("aaaaa");

    CHECK(ids.size() == 3);
    CHECK(tokenizer.decode(ids) == "aaaaa");
}

TEST_CASE("bpe breaks pair frequency ties lexicographically") {
    BPETokenizer tokenizer("bac", 1);

    CHECK(tokenizer.encode("ac").size() == 1);
    CHECK(tokenizer.encode("ba").size() == 2);
}

TEST_CASE("bpe stops when the corpus has no pairs left") {
    BPETokenizer tokenizer("x", 100);

    CHECK(tokenizer.vocab_size() == 256);
    CHECK(tokenizer.encode("xyz").size() == 3);
    CHECK(tokenizer.decode(tokenizer.encode("xyz")) == "xyz");
}

TEST_CASE("bpe handles empty input") {
    BPETokenizer tokenizer("banana", 10);

    CHECK(tokenizer.encode("").empty());
    CHECK(tokenizer.decode({}).empty());
}

TEST_CASE("bpe round trips bytes absent from the training corpus") {
    BPETokenizer tokenizer("banana bandana", 10);
    string text = string("unseen\0byte", 11) + static_cast<char>(255);
    vec<int> ids = tokenizer.encode(text);

    CHECK(tokenizer.decode(ids) == text);
}

TEST_CASE("bpe rejects invalid token ids") {
    BPETokenizer tokenizer("abababab", 2);

    CHECK_THROWS_AS(tokenizer.decode({-1}), std::runtime_error);
    CHECK_THROWS_AS(tokenizer.decode({tokenizer.vocab_size()}), std::runtime_error);
}

TEST_SUITE_END;

#pragma once
#include "util.h"

struct CharTokenizer {
private:
    vec<char> vocab; // vocab[i] = c ---> char c has token id i

public:
    // construct vocab from text in corpus
    CharTokenizer(const string &corpus);

    // return text processed into token ids
    vec<int> encode(const string &text) const;
    // return token ids processed into text
    string decode(const vec<int> &ids) const;

    int vocab_size() const { return sz(vocab); };
};

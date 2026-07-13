#pragma once
#include "util.h"
#include <map>

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

struct BPETokenizer {
private:
    vec<pair<string, string>> steps; // each step says merge all occurrences of adjacent first, second
    vec<string> vocab; // vocab[i] = s ---> string s has token id i
    map<string, int> rev_vocab; // vocab[s] = i ---> string s has token id i

    // rebuild vocab from the learned steps
    void rebuild_vocab();

public:
    // construct an empty tokenizer for loading
    BPETokenizer() = default;
    // construct vocab from text in corpus
    BPETokenizer(const string &corpus, int steps);

    // serialize learned steps as quoted string pairs
    string save() const;
    // deserialize learned steps from quoted string pairs
    static BPETokenizer load(const string &text);

    // return text processed into token ids
    vec<int> encode(const string &text) const;
    // return token ids processed into text
    string decode(const vec<int> &ids) const;

    int vocab_size() const { return sz(vocab); };
};

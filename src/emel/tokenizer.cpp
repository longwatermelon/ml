#include "tokenizer.h"
#include <stdexcept>

// ---- char tokenizer ----

// construct vocab from text in corpus
CharTokenizer::CharTokenizer(const string &corpus) {
    bool seen[256] = {};
    for (unsigned char c : corpus) {
        seen[c] = true;
    }

    for (int c = 0; c < 256; ++c) {
        if (seen[c]) {
            vocab.push_back(c);
        }
    }
}

// return text processed into token ids
vec<int> CharTokenizer::encode(const string &text) const {
    vec<int> result;
    for (char c : text) {
        // find index in vocab
        int ind = -1;
        for (int i = 0; i < sz(vocab); ++i) {
            if (vocab[i] == c) {
                ind = i;
                break;
            }
        }

        // not present in vocab?
        if (ind == -1) {
            throw std::runtime_error("char not in vocab");
        }

        // push
        result.push_back(ind);
    }

    return result;
}

// return token ids processed into text
string CharTokenizer::decode(const vec<int> &ids) const {
    string result;
    // map ids to chars
    for (int id : ids) {
        // out of range?
        if (id < 0 || id >= sz(vocab)) {
            throw std::runtime_error("id not recognized in vocab");
        }

        // push
        result.push_back(vocab[id]);
    }

    return result;
}

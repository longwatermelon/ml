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

// ---- BPE tokenizer ----

static vec<string> merge_lr(const vec<string> &text, const string &left, const string &right) {
    vec<string> out;
    for (int i = 0; i < sz(text); ++i) {
        if (i+1 < sz(text) && text[i] == left && text[i+1] == right) {
            // matched, merge
            out.push_back(left + right);
            i++;
        } else {
            // no match, don't merge
            out.push_back(text[i]);
        }
    }

    return out;
}

// construct vocab from text in corpus
BPETokenizer::BPETokenizer(const string &corpus, int steps) {
    set<string> all_chunks;
    for (int c = 0; c < 256; ++c) {
        all_chunks.insert(string(1,c));
    }

    // construct char-split string
    vec<string> split_str;
    for (char c : corpus) {
        split_str.push_back(string(1,c));
    }

    // run bpe steps
    for (int step = 0; step < steps; ++step) {
        // track occurrences of every pair
        map<pair<string, string>, int> occ;
        for (int i = 0; i < sz(split_str) - 1; ++i) {
            occ[{split_str[i], split_str[i+1]}]++;
        }
        if (occ.empty()) break;

        // get pair with max occurrences
        int mx_cnt = 0;
        pair<string, string> best_p;
        for (auto &[p, cnt] : occ) {
            if (cnt > mx_cnt) {
                mx_cnt = cnt;
                best_p = p;
            }
        }

        // merge pair & record
        split_str = merge_lr(split_str, best_p.first, best_p.second);
        all_chunks.insert(best_p.first + best_p.second);
        this->steps.push_back(best_p);
    }

    // build vocab with whatever we end up with
    vocab = vec<string>(all(all_chunks));
    for (int i = 0; i < sz(vocab); ++i) {
        rev_vocab[vocab[i]] = i;
    }
}

// return text processed into token ids
vec<int> BPETokenizer::encode(const string &text) const {
    // construct char-split string
    vec<string> split_str;
    for (char c : text) {
        split_str.push_back(string(1,c));
    }

    // run learned merges
    for (auto &[left, right] : steps) {
        split_str = merge_lr(split_str, left, right);
    }

    // convert to token ids
    vec<int> ids;
    for (const string &s : split_str) {
        ids.push_back(rev_vocab.at(s));
    }

    return ids;
}

// return token ids processed into text
string BPETokenizer::decode(const vec<int> &ids) const {
    string out;
    for (int id : ids) {
        // id out of range?
        if (id < 0 || id >= sz(vocab)) {
            throw std::runtime_error("id not recognized in vocab");
        }

        // push to result
        out += vocab[id];
    }
    return out;
}

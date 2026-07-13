#include "tokenizer.h"
#include <cctype>
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

// split text into maximal-length chunks whose characters are of same byte class
static vec<string> pretokenize(const string &text) {
    vec<string> chunks;
    string cur;
    int last_class = -1;

    // iter over chars
    for (char c : text) {
        // identify char class
        unsigned char byte = c;
        int cls = std::isalpha(byte) ? 0
            : std::isdigit(byte) ? 1
            : std::isspace(byte) ? 2
            : 3;

        // push chunk if class mismatch
        if (!cur.empty() && cls != last_class) {
            chunks.push_back(cur);
            cur.clear();
        }

        // append to current chunk
        cur.push_back(c);
        last_class = cls;
    }

    // push last untracked chunk
    if (!cur.empty()) {
        chunks.push_back(cur);
    }

    return chunks;
}

// construct vocab from text in corpus
BPETokenizer::BPETokenizer(const string &corpus, int steps) {
    // count chunk occurrences
    map<string, int> counts;
    vec<string> pretok_chunks = pretokenize(corpus);
    for (const string &chunk : pretok_chunks) {
        counts[chunk]++;
    }

    // run bpe on chunks: split into character-length runs initially
    vec<pair<vec<string>, int>> chunks; // list of [split string, # occurrences]
    for (const auto &[chunk, count] : counts) {
        vec<string> split_chunk;
        for (char c : chunk) {
            split_chunk.push_back(string(1,c));
        }
        chunks.push_back({split_chunk, count});
    }

    // track all existing merged character runs as we learn bpe
    // start with individual characters
    set<string> all_runs;
    for (int c = 0; c < 256; ++c) {
        all_runs.insert(string(1,c));
    }

    // learn merges per chunk (contained between pretoken boundaries)
    for (int step = 0; step < steps; ++step) {
        // count all occurrences of all adjacent run pairs [left, right] across all pretokens
        map<pair<string, string>, int> occ;
        for (auto &[chunk, cnt] : chunks) {
            for (int i = 0; i+1 < sz(chunk); ++i) {
                occ[{chunk[i], chunk[i+1]}] += cnt;
            }
        }
        if (occ.empty()) {
            break;
        }

        // find highest count of [left, right] run pairs
        int best_cnt = -1;
        pair<string, string> best_p;
        for (auto &[p, cnt] : occ) {
            if (cnt > best_cnt) {
                best_cnt = cnt;
                best_p = p;
            }
        }

        // merge [left, right] in all chunks that it exists in
        for (auto &[chunk, cnt] : chunks) {
            chunk = merge_lr(chunk, best_p.first, best_p.second);
        }
        this->steps.push_back(best_p);

        // track in all runs that have existed
        all_runs.insert(best_p.first + best_p.second);
    }

    // populate vocab with all runs we've encountered through learning bpe
    for (auto &s : all_runs) {
        vocab.push_back(s);
        rev_vocab[s] = sz(vocab) - 1;
    }
}

// return text processed into token ids
vec<int> BPETokenizer::encode(const string &text) const {
    vec<int> ids;
    for (const string &chunk : pretokenize(text)) {
        // split the chunk into byte tokens
        vec<string> split_chunk;
        for (char c : chunk) {
            split_chunk.push_back(string(1, c));
        }

        // apply learned merges within this chunk
        for (const auto &[left, right] : steps) {
            split_chunk = merge_lr(split_chunk, left, right);
        }

        for (const string &token : split_chunk) {
            ids.push_back(rev_vocab.at(token));
        }
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

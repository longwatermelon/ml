#pragma once
#include "emel/util.h"
#include "emel/nn.h"
#include <sstream>

// read contents from file into string
inline string read_file(const string &path) {
    std::ifstream ifs(path);
    std::stringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

// GPT module
struct GPT : nn::Module {
    // hyperparams
    int d,h,N,d_ff,V;

    // modules
    nn::Embedding embedding, pos_enc;
    vec<nn::TransformerBlock> blocks;
    nn::LayerNorm ln;
    nn::Linear unembedding;

    // ctor
    GPT(int Tmax, int d, int h, int N, int d_ff, int V)
        : d(d), h(h), N(N), d_ff(d_ff), V(V),
          embedding(V,d), pos_enc(Tmax,d), ln(d), unembedding(d,V) {
        for (int i = 0; i < N; ++i) {
            blocks.push_back(nn::TransformerBlock(d, h, d_ff));
            blocks.back().with_causal_mask();
        }
    }

    // forward pass
    GTensor forward(const GTensor &X) override {
        GTensor Xp = X;

        // embedding + pos. enc
        vec<int> shape = Xp.get_tensor().shape;
        GTensor P(shape, 0.f);
        for (int b = 0; b < shape[0]; ++b) {
            for (int t = 0; t < shape[1]; ++t) {
                P.get_tensor_ref().at({b,t}) = t;
            }
        }
        Xp = embedding.forward(Xp) + pos_enc.forward(P);

        // transformer
        for (auto &block : blocks) {
            Xp = block.forward(Xp);
        }

        // LN + unembed
        Xp = ln.forward(Xp);
        Xp = unembedding.forward(Xp);

        return Xp;
    }

    // params
    vec<GTensor*> params() override {
        vec<GTensor*> res;
        auto add_params = [&](nn::Module *mod) {
            vec<GTensor*> params = mod->params();
            res.insert(end(res), all(params));
        };
        add_params(&embedding);
        add_params(&pos_enc);
        for (auto &block : blocks) {
            add_params(&block);
        }
        add_params(&ln);
        add_params(&unembedding);
        return res;
    }
};

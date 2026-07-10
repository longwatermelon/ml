#include "emel/nn.h"
#include "emel/tokenizer.h"
#include <sstream>

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
        GTensor P(shape, 0.);
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

int main() {
    int T = 32;
    int Tmax = 32;
    int d = 64;
    int h = 4;
    int N = 2;
    int d_ff = 4*d;
    int B = 8;
    double lr = 3e-4;
    int Mtrain = 4096; // # train windows
    int Mtest = 256; // # test windows

    // read text corpus
    std::ifstream ifs("data/shakespeare/input.txt");
    std::stringstream ss;
    ss << ifs.rdbuf();
    string corpus = ss.str();

    // tokenize corpus
    CharTokenizer tokz(corpus);
    int V = tokz.vocab_size(); // vocab size
    vec<int> corpus_toks = tokz.encode(corpus);

    // split corpus into 90% train 10% test
    int split_ind = 0.9 * sz(corpus_toks);
    vec<int> toks_train(begin(corpus_toks), begin(corpus_toks) + split_ind);
    vec<int> toks_test(begin(corpus_toks) + split_ind, end(corpus_toks));

    // prepare examples: select random subarrays of len T+1
    auto populate_examples = [&](Tensor &X, Tensor &Y, int M, const vec<int> &toks) {
        for (int i = 0; i < M; ++i) {
            int st = rand() % (sz(toks) - T);
            for (int j = 0; j < T+1; ++j) {
                // X
                if (j < T) {
                    X.at({i, j}) = toks[st+j];
                }

                // Y
                if (j-1 >= 0) {
                    Y.at({i, j-1, toks[st+j]}) = 1.;
                }
            }
        }
    };
    Tensor Xtrain({Mtrain,T}, 0.), Ytrain({Mtrain,T,V}, 0.);
    populate_examples(Xtrain, Ytrain, Mtrain, toks_train);
    Tensor Xtest({Mtest,T}, 0.), Ytest({Mtest,T,V}, 0.);
    populate_examples(Xtest, Ytest, Mtest, toks_test);

    // build model
    GPT model(Tmax, d, h, N, d_ff, V);
    Adam opt(model.params(), lr);
    nn::train(model, Xtrain, Ytrain, 2, Loss::CrossEntropyLogits, opt, B);
}

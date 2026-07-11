#include "util.h"
#include "emel/nn.h"
#include "emel/tokenizer.h"
#include <random>

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

const int T = 32;
const int Tmax = 32;
const int d = 64;
const int h = 4;
const int N = 4;
const int d_ff = 4*d;
const int B = 8;
const double lr = 3e-4;
const int Mtrain = 4096*5; // # train windows
const int Mtest = 256; // # test windows
int V;

CharTokenizer build_tokenizer(const string &filename) {
    string corpus = read_file(filename);

    // tokenize corpus
    CharTokenizer tokz(corpus);
    V = tokz.vocab_size(); // vocab size

    return tokz;
}

// train model
void train(const string &out_path) {
    // tokenize corpus
    string input_path = "data/shakespeare/input.txt";
    CharTokenizer tokz = build_tokenizer(input_path);
    string corpus = read_file(input_path);
    vec<int> corpus_toks = tokz.encode(corpus);

    // split corpus into 90% train 10% test
    int split_ind = 0.9 * sz(corpus_toks);
    vec<int> toks_train(begin(corpus_toks), begin(corpus_toks) + split_ind);
    vec<int> toks_test(begin(corpus_toks) + split_ind, end(corpus_toks));

    // prepare examples: select random subarrays of len Tmax+1
    auto populate_examples = [&](Tensor &X, Tensor &Y, int M, const vec<int> &toks) {
        for (int i = 0; i < M; ++i) {
            int st = rand() % (sz(toks) - Tmax);
            for (int j = 0; j < Tmax+1; ++j) {
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
    Tensor Xtrain({Mtrain,Tmax}, 0.), Ytrain({Mtrain,Tmax,V}, 0.);
    populate_examples(Xtrain, Ytrain, Mtrain, toks_train);
    Tensor Xtest({Mtest,Tmax}, 0.), Ytest({Mtest,Tmax,V}, 0.);
    populate_examples(Xtest, Ytest, Mtest, toks_test);

    // build model
    GPT model(Tmax, d, h, N, d_ff, V);
    Adam opt(model.params(), lr);
    nn::train(model, Xtrain, Ytrain, 4, Loss::CrossEntropyLogits, opt, B);

    // save
    vec<uint8_t> bytes = nn::save(model);
    write_file_bytes(out_path, bytes);
    printf("saved model to path '%s'\n", out_path.c_str());
}

// load model from path, run inference
void inference(const string &in_path) {
    // build tokenizer
    CharTokenizer tokz = build_tokenizer("data/shakespeare/input.txt");

    // load model
    GPT model(Tmax, d, h, N, d_ff, V);
    nn::load(model, read_file_bytes(in_path));

    // generation
    string prompt = "ROMEO: ";
    int gen_count = 100;
    double temp = 0.8;
    vec<int> toks = tokz.encode(prompt);
    std::mt19937 rng(std::random_device{}());

    for (int step = 0; step < gen_count; ++step) {
        // retain only the available context window
        int ctx_len = min(Tmax, sz(toks));
        int ctx_start = sz(toks) - ctx_len;

        Tensor X({1, ctx_len}, 0.);
        for (int t = 0; t < ctx_len; ++t) {
            X.at({0, t}) = toks[ctx_start + t];
        }

        // inference
        Tensor logits = model.forward(X).get_tensor();

        // to prob distribution
        logits.ediv(Tensor({1}, temp));
        Tensor S = logits.softmax(2);
        vec<double> weights(V);
        for (int i = 0; i < V; ++i) {
            weights[i] = S.at({0, ctx_len - 1, i});
        }

        // sample from prob distribution
        std::discrete_distribution<int> sample(all(weights));
        toks.push_back(sample(rng));
    }

    printf("%s\n", tokz.decode(toks).c_str());
}

int main(int argc, char **argv) {
    train("shakespeare.bin");
    // inference("shakespeare.bin");
}

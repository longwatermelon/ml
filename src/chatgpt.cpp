#include "util.h"
#include "emel/nn.h"
#include "emel/tokenizer.h"
#include <random>
#include <filesystem>

const int T = 256;
const int Tmax = 256;
const int d = 128;
const int h = 4;
const int N = 6;
const int d_ff = 4*d;
const int B = 32;
const float lr = 1e-3f;
const int Mtrain = 24000; // # train windows
const int Mtest = 1024; // # test windows
int V;

CharTokenizer build_tokenizer(const string &filename) {
    string corpus = read_file(filename);

    // tokenize corpus
    CharTokenizer tokz(corpus);
    V = tokz.vocab_size(); // vocab size

    return tokz;
}

// train model
void train(const string &out_path, int epochs, const string &model_path = "") {
    // tokenize corpus
    string input_path = "data/dailydialog/input.txt";
    CharTokenizer tokz = build_tokenizer(input_path);
    string corpus = read_file(input_path);
    vec<int> corpus_toks = tokz.encode(corpus);

    // split corpus into 90% train 10% test
    int split_ind = 0.9f * sz(corpus_toks);
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
                    Y.at({i, j-1}) = toks[st+j];
                }
            }
        }
    };
    Tensor Xtrain({Mtrain,Tmax}, 0.f), Ytrain({Mtrain,Tmax}, 0.f);
    populate_examples(Xtrain, Ytrain, Mtrain, toks_train);
    Tensor Xtest({Mtest,Tmax}, 0.f), Ytest({Mtest,Tmax}, 0.f);
    populate_examples(Xtest, Ytest, Mtest, toks_test);

    // build model
    GPT model(Tmax, d, h, N, d_ff, V);
    if (model_path != "") {
        nn::load(model, read_file_bytes(model_path));
    }

    Adam opt(model.params(), lr);
    nn::train(model, Xtrain, Ytrain, epochs, Loss::CrossEntropyLogitsSparse,
              opt, B, &Xtest, &Ytest);

    // save
    vec<uint8_t> bytes = nn::save(model);
    write_file_bytes(out_path, bytes);
    printf("saved model to path '%s'\n", out_path.c_str());
}

int next_token(GPT &model, const vec<int> &toks, double temp) {
    static std::mt19937 rng(std::random_device{}());

    // retain only the available context window
    int ctx_len = min(Tmax, sz(toks));
    int ctx_start = sz(toks) - ctx_len;

    Tensor X({1, ctx_len}, 0.f);
    for (int t = 0; t < ctx_len; ++t) {
        X.at({0, t}) = toks[ctx_start + t];
    }

    // inference
    Tensor logits = model.forward(X).get_tensor();

    // to prob distribution
    logits = logits.ediv(Tensor({1}, temp));
    Tensor S = logits.softmax(2);
    vec<float> weights(V);
    for (int i = 0; i < V; ++i) {
        weights[i] = S.at({0, ctx_len - 1, i});
    }

    // sample from prob distribution
    std::discrete_distribution<int> sample(all(weights));
    return sample(rng);
}

// load model from path, run inference
void inference(const string &in_path) {
    // build tokenizer
    CharTokenizer tokz = build_tokenizer("data/dailydialog/input.txt");

    // load model
    GPT model(Tmax, d, h, N, d_ff, V);
    nn::load(model, read_file_bytes(in_path));

    // generation
    vec<int> context;
    while (true) {
        // get user prompt
        string user_prompt;
        std::cout << "> ";
        std::getline(std::cin, user_prompt);

        // insert user tag + prompt to context
        vec<int> user_tag = tokz.encode("USER:\n");
        context.insert(end(context), all(user_tag));
        vec<int> toks = tokz.encode(user_prompt);
        context.insert(end(context), all(toks));
        vec<int> bot_tag = tokz.encode("\nBOT:\n");
        context.insert(end(context), all(bot_tag));

        // generate response
        float temp = 0.7f;
        for (int step = 0; step < 1000; ++step) {
            int next_tok = next_token(model, context, temp);
            context.push_back(next_tok);

            char ch = tokz.decode({next_tok})[0];
            putchar(ch);
            if (ch == '\n') {
                break;
            }
        }
    }
}

int main(int argc, char **argv) {
    string model_path = "chatgpt.bin";
    if (argc > 1 && strcmp(argv[1], "train") == 0) {
        // train
        for (int i = 0; i < 4; ++i) {
            string load_path;
            if (std::filesystem::exists(model_path)) {
                load_path = model_path;
            }
            train(model_path, 1, load_path);
        }
    } else {
        // inference
        inference(model_path);
    }
}

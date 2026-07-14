#include "util.h"
#include "emel/nn.h"
#include "emel/tokenizer.h"
#include <random>
#include <filesystem>

const int T = 384;
const int Tmax = 384;
const int d = 256;
const int h = 8;
const int N = 6;
const int d_ff = 4*d;
const int B = 16;
const float lr = 1e-3f;
const int Mtrain = 24000; // # train windows
const int Mtest = 1024; // # test windows
const string input_path = "data/tinystories/input.txt";
const string model_path = "models/tinystories.bin";
const string tokz_path = "models/tinystories.tokz";
int V;

BPETokenizer build_tokenizer(const string &filename) {
    string corpus = read_file(filename);

    // tokenize corpus
    BPETokenizer tokz(corpus, 2000);
    V = tokz.vocab_size(); // vocab size

    return tokz;
}

// train model
void train(int epochs, bool load_checkpoint, bool randomize_windows) {
    // tokenize corpus
    BPETokenizer tokz = BPETokenizer::load(read_file(tokz_path));
    V = tokz.vocab_size();
    string corpus = read_file(input_path);
    printf("tokenizing corpus...\n");
    vec<int> corpus_toks = tokz.encode(corpus);
    printf("tokenized!\n");

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
    if (load_checkpoint) {
        nn::load(model, read_file_bytes(model_path));
    }

    Adam opt(model.params(), lr);

    // train
    for (int i = 0; i < epochs; ++i) {
        printf("==== ROUND %d/%d ====\n", i+1, epochs);
        nn::train(model, Xtrain, Ytrain, 1, Loss::CrossEntropyLogitsSparse,
                  opt, B, &Xtest, &Ytest);

        // save checkpoint
        vec<uint8_t> bytes = nn::save(model);
        write_file_bytes(model_path, bytes);
        printf("saved model to path '%s'\n", model_path.c_str());

        // randomize windows in training data?
        if (randomize_windows && i+1 < epochs) {
            populate_examples(Xtrain, Ytrain, Mtrain, toks_train);
        }
    }
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
void inference() {
    // load tokenizer
    BPETokenizer tokz = BPETokenizer::load(read_file(tokz_path));
    V = tokz.vocab_size();

    // load model
    GPT model(Tmax, d, h, N, d_ff, V);
    nn::load(model, read_file_bytes(model_path));

    // generate story
    string prefill = "Once upon a time, ";
    vec<int> context = tokz.encode(prefill);
    printf("%s", prefill.c_str());
    fflush(stdout);
    float temp = 0.7f;
    for (int step = 0; step < 500; ++step) {
        int next_tok = next_token(model, context, temp);
        context.push_back(next_tok);

        string s = tokz.decode({next_tok});
        printf("%s", s.c_str());
        fflush(stdout);
    }
}

int main(int argc, char **argv) {
    srand(time(0));

    // tokenizer first; if not built, build it and save it
    if (!std::filesystem::exists(tokz_path)) {
        printf("generating tokenizer...\n");
        BPETokenizer tokz = build_tokenizer(input_path);
        string contents = tokz.save();

        std::ofstream ofs(tokz_path);
        ofs << contents;
        ofs.close();
        printf("tokenizer built!\n");
    }

    // train model or inference on existing model?
    if (argc > 1 && strcmp(argv[1], "train") == 0) {
        // train
        bool load_checkpoint = std::filesystem::exists(model_path);
        train(20, load_checkpoint, true);
    } else {
        // inference
        inference();
    }
}

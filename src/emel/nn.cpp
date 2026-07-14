#include "nn.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <chrono>

namespace nn {

// fan-in random weight initialization
static void init_fan_in(GTensor &W, int fan_in) {
    float limit = sqrt(6.f / fan_in);
    W.get_tensor_ref().apply_inplace([&](float) {
        return ((float)rand() / RAND_MAX * 2.f - 1.f) * limit;
    });
}

// ---- model training ----

static Tensor select_minibatch(int st, int cnt, const Tensor &X, vec<int> ord) {
    vec<int> minibatch_shape = X.shape;
    minibatch_shape[0] = cnt;
    Tensor Xb(minibatch_shape, 0.f);
    for (int i = 0; i < cnt; ++i) {
        int ind = ord[st+i];
        vec<int> cur(sz(X.shape), 0);
        cur[0] = ind;
        vec<int> lim = X.shape;
        lim[0] = ind;
        do {
            vec<int> bcur = cur;
            bcur[0] = i;
            Xb.at(bcur) = X.at(cur);
        } while (advance_ind(cur, lim));
    }

    return Xb;
}

// evaluate average loss in minibatches to limit memory use
static float eval_loss(Module &model, const Tensor &X, const Tensor &Y,
                       Loss loss, int batch_size) {
    int m = X.shape[0];
    vec<int> inds(m);
    iota(all(inds), 0);

    float total_loss = 0.f;
    for (int st = 0; st < m; st += batch_size) {
        int cur_batch = min(batch_size, m-st);
        Tensor Xb = select_minibatch(st, cur_batch, X, inds);
        Tensor Yb = select_minibatch(st, cur_batch, Y, inds);
        total_loss += apply_loss_scalar(model.forward(Xb), Yb, loss) * cur_batch;
    }
    return total_loss / m;
}

// train a model and optionally report test loss after each epoch
void train(Module &model, const Tensor &X, const Tensor &Y, int epochs, Loss loss,
           Optimizer &opt, int batch_size, const Tensor *Xtest, const Tensor *Ytest) {
    assert(batch_size > 0);
    assert((Xtest == nullptr) == (Ytest == nullptr));
    if (Xtest != nullptr) {
        assert(Xtest->shape[0] > 0 && Xtest->shape[0] == Ytest->shape[0]);
    }

    int m = X.shape[0];
    std::mt19937 g(0);

    auto st = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // minibatching - process in chunks of batch_size
        // shuffle data order first
        vec<int> inds(m);
        iota(all(inds), 0);
        shuffle(all(inds), g);

        float avg_loss = 0.f;
        int minibatch_ind = 0;
        int tot_minibatches = (m+batch_size-1) / batch_size;
        for (int st = 0; st < m; st += batch_size) {
            printf("\repoch %d: minibatch %d/%d...", epoch+1, minibatch_ind+1, tot_minibatches);
            fflush(stdout);

            int cur_batch = min(batch_size, m-st);
            Tensor Xb = select_minibatch(st, cur_batch, X, inds);
            Tensor Yb = select_minibatch(st, cur_batch, Y, inds);

            // forward pass
            GTensor Yhatb = model.forward(Xb);

            // diagnostic loss for reporting later
            avg_loss += apply_loss_scalar(Yhatb, Yb, loss) * cur_batch;

            // backward pass
            GTensor g_loss = apply_loss(Yhatb, Yb, loss);
            g_loss.compute_all_grads();
            opt.step();

            minibatch_ind++;
        }
        avg_loss /= m;

        // eval
        printf("\repoch %d/%d done | train loss = %.6f", epoch+1, epochs, avg_loss);
        if (Xtest != nullptr) {
            float test_loss = eval_loss(model, *Xtest, *Ytest, loss, batch_size);
            printf(" | test loss = %.6f", test_loss);
        }
        putchar('\n');
        fflush(stdout);
    }
    putchar('\n');

    int diff = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - st).count();
    printf("%dh %dm elapsed\n", diff/3600, (diff%3600)/60);
}

// ---- model save/load ----

// serialize model params (params only, no architecture) to bytes
vec<uint8_t> save(Module &model) {
    vec<uint8_t> bytes;

    vec<GTensor*> ps = model.params();
    append_bytes(bytes, (uint32_t)sz(ps));
    for (GTensor *p : ps) {
        vec<uint8_t> t_bytes = p->get_tensor().serialize();
        bytes.insert(bytes.end(), all(t_bytes));
    }

    return bytes;
}

// load serialized params into an already-constructed model of the same architecture
void load(Module &model, const vec<uint8_t> &bytes) {
    size_t pos = 0;

    vec<GTensor*> ps = model.params();
    uint32_t n = read_bytes<uint32_t>(bytes, pos);
    assert((int)n == sz(ps));

    for (GTensor *p : ps) {
        Tensor t = Tensor::deserialize(bytes, pos);
        assert(t.shape == p->get_tensor().shape);
        p->get_tensor_ref() = t;
    }
    assert(pos == bytes.size());
}

// ---- linear module ----

// ctor
Linear::Linear(int n_prev, int n) {
    W = GTensor({n_prev, n}, 0.f);
    b = GTensor({n}, 0.f);

    init_fan_in(W, n_prev);
}

// forward pass
GTensor Linear::forward(const GTensor &A_prev) {
    return A_prev * W + b;
}

// params
vec<GTensor*> Linear::params() {
    return {&W, &b};
}

// ---- relu module ----

// forward pass
GTensor Relu::forward(const GTensor &A_prev) {
    return A_prev.relu();
}

// params
vec<GTensor*> Relu::params() {
    return {};
}

// ---- softmax module ----

// forward pass
GTensor Softmax::forward(const GTensor &A_prev) {
    GTensor argmax = A_prev.max_reduce(1, true);
    GTensor numerator = (A_prev - argmax).exp();
    GTensor sum = numerator.sum_reduce(1, true);
    GTensor result = numerator.ediv(sum);
    return result;
}

// params
vec<GTensor*> Softmax::params() {
    return {};
}

// ---- sequential module ----

// forward pass
GTensor Sequential::forward(const GTensor &A_prev) {
    GTensor in = A_prev;
    for (int i = 0; i < sz(layers); ++i) {
        GTensor out = layers[i]->forward(in);
        in = out;
    }

    return in;
}

// params
vec<GTensor*> Sequential::params() {
    vec<GTensor*> out;
    for (int i = 0; i < sz(layers); ++i) {
        vec<GTensor*> layer_params = layers[i]->params();
        out.insert(end(out), all(layer_params));
    }

    return out;
}

// ---- conv2d module ----

// ctor
Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size) {
    W = GTensor({
        in_channels * kernel_size * kernel_size,
        out_channels
    }, 0.f);
    b = GTensor({out_channels}, 0.f);
    k = kernel_size;

    init_fan_in(W, in_channels * kernel_size * kernel_size);
}

// forward pass
GTensor Conv2d::forward(const GTensor &X) {
    // dimensions
    int N = X.get_tensor().shape[0];
    int Cin = X.get_tensor().shape[1];
    int H = X.get_tensor().shape[2];
    int W = X.get_tensor().shape[3];
    int Hp = H-k+1, Wp = W-k+1;
    int Cout = this->b.get_tensor().shape[0];

    // batched
    auto im2col = [&](const GTensor &X) {
        GTensor Xr = X.reshape({N, Cin * H * W});
        Tensor I({N, Hp*Wp, Cin*k*k, 2}, 0.f);
        for (int b = 0; b < N; ++b) {
            for (int i = 0; i < Hp; ++i) {
                for (int j = 0; j < Wp; ++j) {
                    // example b, output pixel (i,j), top-left of kernel
                    for (int c = 0; c < Cin; ++c) {
                        for (int oi = i; oi < i+k; ++oi) {
                            for (int oj = j; oj < j+k; ++oj) {
                                I.at({b, i*Wp+j, c*k*k + (oi-i)*k + (oj-j), 0}) = b;
                                I.at({b, i*Wp+j, c*k*k + (oi-i)*k + (oj-j), 1}) = c*H*W + oi*W + oj;
                            }
                        }
                    }
                }
            }
        }

        GTensor Xp = Xr.gather(I);
        return Xp;
    };

    GTensor Xp = im2col(X);
    GTensor wxb = Xp * this->W + b;
    GTensor out = wxb.permute({0,2,1}).reshape({N, Cout, Hp, Wp});
    return out;
}

// params
vec<GTensor*> Conv2d::params() {
    return {&W,&b};
}

// ---- flatten module ----

// forward pass
GTensor Flatten::forward(const GTensor &A_prev) {
    vec<int> shape = A_prev.get_tensor().shape;
    int prod = 1;
    for (int i = 1; i < sz(shape); ++i) {
        prod *= shape[i];
    }
    vec<int> new_shape = {shape[0], prod};
    return A_prev.reshape(new_shape);
}

// params
vec<GTensor*> Flatten::params() {
    return {};
}

// ---- attention module ----

// ctor
Attention::Attention(int d, int d_k, int d_v) {
    this->d = d;
    this->d_k = d_k;
    this->d_v = d_v;
    W_Q = GTensor({d, d_k}, 0.f);
    W_K = GTensor({d, d_k}, 0.f);
    W_V = GTensor({d, d_v}, 0.f);

    init_fan_in(W_Q, d);
    init_fan_in(W_K, d);
    init_fan_in(W_V, d);
}

// forward pass
GTensor Attention::forward(const GTensor &X) {
    GTensor Q = X * W_Q;
    GTensor K = X * W_K;

    // calculate S
    GTensor Snum = Q * K.transpose();
    GTensor Sdenom = GTensor({1}, sqrt((float)d_k));
    GTensor Slogits = Snum.ediv(Sdenom);
    // causal mask?
    if (causal_mask) {
        int n = sz(X.get_tensor().shape);
        int T = X.get_tensor().shape[n-2];
        Tensor M({T,T}, 0.f);
        for (int i = 0; i < T; ++i) {
            for (int j = i+1; j < T; ++j) {
                M.at({i,j}) = -std::numeric_limits<float>::infinity();
            }
        }

        Slogits = Slogits + GTensor(M);
    }
    // logits -> probs
    int last = sz(Slogits.get_tensor().shape) - 1;
    GTensor S = Slogits.softmax(last);

    // V
    GTensor V = X * W_V;
    GTensor Xp = S * V;

    return Xp;
}

// params
vec<GTensor*> Attention::params() {
    return {&W_Q, &W_K, &W_V};
}

// ---- multi-head attention module ----

// ctor (d must be divisible by h)
MultiHeadAttention::MultiHeadAttention(int d, int h) {
    assert(h != 0 && d % h == 0);

    this->d = d;
    this->h = h;

    // init params
    W_O.resize(h);
    for (int i = 0; i < h; ++i) {
        heads.push_back(Attention(d, d/h, d/h));
        W_O[i] = GTensor({d/h, d}, 0.f);
        init_fan_in(W_O[i], d);
    }
}

// forward pass
GTensor MultiHeadAttention::forward(const GTensor &X) {
    vec<GTensor> head_res(h);
    for (int i = 0; i < h; ++i) {
        head_res[i] = heads[i].forward(X) * W_O[i];
    }

    GTensor Xp = head_res[0];
    for (int i = 1; i < h; ++i) {
        Xp = Xp + head_res[i];
    }
    return Xp;
}

// params
vec<GTensor*> MultiHeadAttention::params() {
    vec<GTensor*> res;
    for (int i = 0; i < h; ++i) {
        vec<GTensor*> head_params = heads[i].params();
        res.insert(end(res), all(head_params));
        res.push_back(&W_O[i]);
    }
    return res;
}

// ---- layernorm module ----

// ctor
LayerNorm::LayerNorm(int d) {
    this->d = d;
    gamma = GTensor({d}, 1.f);
    beta = GTensor({d}, 0.f);
}

// forward pass
GTensor LayerNorm::forward(const GTensor &X) {
    vec<int> shape = X.get_tensor().shape;
    int n = sz(shape);
    assert(n > 0 && shape[n-1] == d);

    // mean, var
    GTensor mean = X.sum_reduce(n-1, true).ediv(GTensor({1}, shape[n-1]));
    GTensor diff = X - mean;
    GTensor var = diff.hadamard(diff).sum_reduce(n-1, true).ediv(GTensor({1}, shape[n-1]));

    // compute layernorm
    float eps = 1e-5f;
    GTensor frac = diff.ediv((var + GTensor({1}, eps)).sqrt());
    GTensor result = gamma.hadamard(frac) + beta;

    return result;
}

// params
vec<GTensor*> LayerNorm::params() {
    return {&gamma, &beta};
}

// ---- embedding module ----

// ctor
Embedding::Embedding(int vocab, int d) {
    v = vocab;
    this->d = d;
    W = GTensor({v, d}, 0.f);
    W.get_tensor_ref().apply_inplace([](float x){return (float)rand() / RAND_MAX * 0.04f - 0.02f;});
}

// forward pass: map X of indices of shape [...] -> [..., d], replacing indices with vectors
GTensor Embedding::forward(const GTensor &X) {
    // setup
    const Tensor &t = X.get_tensor();
    vec<int> shape = t.shape;
    vec<int> new_shape = shape;
    new_shape.push_back(d);
    new_shape.push_back(2);

    // index tensor
    Tensor I(new_shape, 0.f);
    vec<int> cur(sz(shape), 0);
    vec<int> lim = shape;
    do {
        int id = (int)t.at(cur);
        vec<int> ind = cur;
        ind.push_back(0); // d
        ind.push_back(0); // 0 or 1, index tuple
        for (int i = 0; i < d; ++i) {
            ind[sz(ind)-2] = i;
            ind.back() = 0; I.at(ind) = id;
            ind.back() = 1; I.at(ind) = i;
        }
    } while (advance_ind(cur, lim));

    // gather
    GTensor res = W.gather(I);
    return res;
}

// params
vec<GTensor*> Embedding::params() {
    return {&W};
}

// ---- transformer module ----

// ctor
TransformerBlock::TransformerBlock(int d, int heads, int d_ff)
    : attn(d, heads), ln_attn(d), ln_mlp(d) {
    this->d = d;
    this->h = heads;
    this->d_ff = d_ff;

    mlp.add<Linear>(d, d_ff);
    mlp.add<Relu>();
    mlp.add<Linear>(d_ff, d);
}

// forward pass
GTensor TransformerBlock::forward(const GTensor &X) {
    GTensor res = X;
    res = res + attn.forward(ln_attn.forward(res));
    res = res + mlp.forward(ln_mlp.forward(res));
    return res;
}

// params
vec<GTensor*> TransformerBlock::params() {
    vec<GTensor*> attn_p = attn.params(),
                  ln_attn_p = ln_attn.params(),
                  mlp_p = mlp.params(),
                  ln_mlp_p = ln_mlp.params();
    vec<GTensor*> res = attn_p;
    res.insert(end(res), all(ln_attn_p));
    res.insert(end(res), all(mlp_p));
    res.insert(end(res), all(ln_mlp_p));
    return res;
}

} // namespace nn

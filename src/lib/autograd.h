#pragma once
#include "util.h"
#include "matrix.h"
#include <memory>

namespace autograd {

enum class FnType {
    Matmul,
    Scale,
};

struct Value {
    FnType f_type;
    vec<shared_ptr<Value>> adj;
    Matrix result, grad;

    // compute cached result, assuming adj results are all populated
    void compute_result();

    // compute grad from parent value v using chain rule
    void compute_grad(Value *v);
};

// matmul AB
shared_ptr<Value> calc_matmul(shared_ptr<Value> A, shared_ptr<Value> B);
// scaling matrix A
shared_ptr<Value> calc_scale(shared_ptr<Value> c, shared_ptr<Value> A);

// traverse DAG topologically and compute grads
void compute_grads(shared_ptr<Value> root);

} // namespace autograd

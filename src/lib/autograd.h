#pragma once
#include "util.h"
#include "matrix.h"
#include <memory>

namespace autograd {

enum class FnType {
    Matmul,
};

struct Value {
    FnType f_type;
    vec<shared_ptr<Value>> adj;
    Matrix result, grad;

    // compute cached result, assuming adj results are all populated
    void compute_result();

    // add chain rule contrib to grads of children in adj
    void add_grads();
};

// matmul AB
shared_ptr<Value> calc_matmul(shared_ptr<Value> A, shared_ptr<Value> B);

// traverse DAG topologically and compute grads
void compute_grads(shared_ptr<Value> root);

} // namespace autograd

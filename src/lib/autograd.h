#pragma once
#include "util.h"
#include "tensor.h"
#include <memory>

namespace autograd {

enum class FnType {
    Matmul,
    Add,
    Hadamard,
    Ediv,
    Relu,
    Exp,
    Log,
    SumReduce,
    MaxReduce,
    Leaf, // serves a semantic purpose only, no functional one, since leaves don't have children
};

/*
 * some notes:
 * f_type only matters for computing result from children, and computing children grad from result.
*/

struct Value {
    FnType f_type;
    vec<shared_ptr<Value>> adj;
    Tensor result, grad;

    // reduction-specific data
    int axis = -1;
    bool keepdims = true;

    // ---- ctors ----

    // creates new node which points to existing nodes; compute result in place.
    // if f_type is reduction, make sure to pass axis and keepdims.
    Value(FnType f_type, const vec<shared_ptr<Value>> &adj, int axis = -1, bool keepdims = true);

    // ---- computation ----

    // compute cached result, assuming adj results are all populated
    void compute_result();

    // add chain rule contrib to grads of children in adj
    void add_child_grads();
};

// traverse DAG topologically and compute grads
// root must be scalar. clears all reachable grads to 0 first.
void compute_all_grads(shared_ptr<Value> root);

namespace fns {
    // matmul AB
    shared_ptr<Value> matmul(shared_ptr<Value> A, shared_ptr<Value> B);
    // add A+B
    shared_ptr<Value> add(shared_ptr<Value> A, shared_ptr<Value> B);
    // hadamard A \odot B
    shared_ptr<Value> hadamard(shared_ptr<Value> A, shared_ptr<Value> B);
    // ediv A \oslash B
    shared_ptr<Value> ediv(shared_ptr<Value> A, shared_ptr<Value> B);
    // relu A
    shared_ptr<Value> relu(shared_ptr<Value> A);
    // exp A
    shared_ptr<Value> exp(shared_ptr<Value> A);
    // log A
    shared_ptr<Value> log(shared_ptr<Value> A);
    // sum-reduce A (axis=k)
    shared_ptr<Value> sum_reduce(shared_ptr<Value> A, int axis, bool keepdims);
    // max-reduce A (axis=k)
    shared_ptr<Value> max_reduce(shared_ptr<Value> A, int axis, bool keepdims);
    // leaf
    shared_ptr<Value> leaf(Tensor result);
} // namespace fns

} // namespace autograd

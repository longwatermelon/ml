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

typedef shared_ptr<Value> ValuePtr;

// traverse DAG topologically and compute grads
// root must be scalar. clears all reachable grads to 0 first.
void compute_all_grads(ValuePtr root);

namespace fns {
    // matmul AB
    ValuePtr matmul(ValuePtr A, ValuePtr B);
    // add A+B
    ValuePtr add(ValuePtr A, ValuePtr B);
    // hadamard A \odot B
    ValuePtr hadamard(ValuePtr A, ValuePtr B);
    // ediv A \oslash B
    ValuePtr ediv(ValuePtr A, ValuePtr B);
    // relu A
    ValuePtr relu(ValuePtr A);
    // exp A
    ValuePtr exp(ValuePtr A);
    // log A
    ValuePtr log(ValuePtr A);
    // sum-reduce A (axis=k)
    ValuePtr sum_reduce(ValuePtr A, int axis, bool keepdims);
    // max-reduce A (axis=k)
    ValuePtr max_reduce(ValuePtr A, int axis, bool keepdims);
    // leaf
    ValuePtr leaf(Tensor result);
} // namespace fns

} // namespace autograd

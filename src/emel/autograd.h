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
    Reshape,
    Gather,
    Permute,
    Sqrt,
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

    // reshape-specific data
    vec<int> new_shape = {};

    // gather-specific data
    Tensor gather_I;

    // permute-specific data
    vec<int> permute_p;

    // ---- ctors ----

    // creates new node which points to existing nodes; compute result in place.
    Value(FnType f_type, const vec<shared_ptr<Value>> &adj);

    Value with_reduction(int axis, bool keepdims) {
        this->axis = axis;
        this->keepdims = keepdims;
        return *this;
    }
    Value with_reshape(const vec<int> &new_shape) {
        this->new_shape = new_shape;
        return *this;
    }
    Value with_gather(const Tensor &I) {
        this->gather_I = I;
        return *this;
    }
    Value with_permute(const vec<int> &p) {
        permute_p = p;
        return *this;
    }

    // ---- computation ----

    // compute cached result, assuming adj results are all populated
    void compute_result();

    // add chain rule contrib to grads of children in adj
    void add_child_grads();
};
} // namespace autograd

// lightweight autograd wrapper on a Tensor.
struct GTensor {
private:
    shared_ptr<autograd::Value> value;

public:
    // ---- ctors ----

    GTensor() = default;
    // init with a tensor
    GTensor(const Tensor &val);
    // tensor constructor
    GTensor(const vec<int> &shape, double value);
    // tensor constructor
    GTensor(const vec<double> &data_1d);
    // tensor constructor
    GTensor(const vec2<double> &data_2d);

    // ---- operators ----

    // matmul
    GTensor operator*(const GTensor &o) const;
    // add
    GTensor operator+(const GTensor &o) const;
    // element-wise mul
    GTensor hadamard(const GTensor &o) const;
    // element-wise div
    GTensor ediv(const GTensor &o) const;
    // element-wise relu
    GTensor relu() const;
    // element-wise exp
    GTensor exp() const;
    // element-wise log
    GTensor log() const;
    // sum reduce
    GTensor sum_reduce(int axis, bool keepdims) const;
    // max reduce
    GTensor max_reduce(int axis, bool keepdims) const;
    // negate
    GTensor operator-() const;
    // subtract
    GTensor operator-(const GTensor &o) const;
    // reshape
    GTensor reshape(const vec<int> &new_shape) const;
    // gather
    GTensor gather(const Tensor &I) const;
    // gather, except if this is 1D, we exclude the redundant trailing axis of length 1
    GTensor gather_flat(const Tensor &I) const;
    // permute
    GTensor permute(const vec<int> &p) const;
    // square root
    GTensor sqrt() const;
    // aka permute last two axes
    GTensor transpose() const;
    // softmax, composed of primitives
    GTensor softmax(int axis) const;

    // ---- getters ----

    const Tensor &get_tensor() const { return value->result; }
    const Tensor &get_grad() const { return value->grad; }
    Tensor &get_tensor_ref() { return value->result; }
    Tensor &get_grad_ref() { return value->grad; }

    // ---- autograd ----

    // compute all grads: ∂this/∂reachable
    // traverses DAG topologically. this must be scalar.
    // clears all reachable grads to 0 first.
    void compute_all_grads();
};

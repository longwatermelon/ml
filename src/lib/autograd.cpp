#include "autograd.h"
using namespace autograd;

// ---- ctors ----

// creates new node which points to existing nodes; compute result in place.
// reduction functions MUST pass axis and keepdims - it's asserted.
Value::Value(FnType f_type, const vec<shared_ptr<Value>> &adj, int axis, bool keepdims)
    : f_type(f_type), adj(adj), axis(axis), keepdims(keepdims) {
    compute_result();
}

// ---- computation ----

// compute cached result, assuming adj results are all populated
void Value::compute_result() {
    switch (f_type) {
    case FnType::Add:
        assert(sz(adj) == 2);
        result = adj[0]->result + adj[1]->result;
        break;
    case FnType::Ediv:
        assert(sz(adj) == 2);
        result = adj[0]->result.ediv(adj[1]->result);
        break;
    case FnType::Exp:
        assert(sz(adj) == 1);
        result = adj[0]->result.apply([](double x){return exp(x);});
        break;
    case FnType::Hadamard:
        assert(sz(adj) == 2);
        result = adj[0]->result.hadamard(adj[1]->result);
        break;
    case FnType::Log:
        assert(sz(adj) == 1);
        result = adj[0]->result.apply([](double x){return log(x);});
        break;
    case FnType::Matmul:
        assert(sz(adj) == 2);
        result = adj[0]->result * adj[1]->result;
        break;
    case FnType::Relu:
        assert(sz(adj) == 1);
        result = adj[0]->result.apply([](double x){return max(0., x);});
        break;
    case FnType::SumReduce:
        assert(sz(adj) == 1);
        result = adj[0]->result.sum(axis, keepdims);
        break;
    case FnType::MaxReduce:
        assert(sz(adj) == 1);
        // argmax mask via one-hot argmax odot with A, then reduce-sum to compress the axis
        result = adj[0]->result.argmax(axis).hadamard(adj[0]->result).sum(axis, keepdims);
        break;
    case FnType::Leaf:
        break;
    }
}

// add chain rule contrib to grads of children in adj
void Value::add_child_grads() {
}

// ---- public API ----

// traverse DAG topologically and compute grads
// root must be scalar. clears all reachable grads to 0 first.
void compute_all_grads(shared_ptr<Value> root) {
}

// matmul AB
shared_ptr<Value> fns::matmul(shared_ptr<Value> A, shared_ptr<Value> B) {
}

// add A+B
shared_ptr<Value> fns::add(shared_ptr<Value> A, shared_ptr<Value> B) {
}

// hadamard A \odot B
shared_ptr<Value> fns::hadamard(shared_ptr<Value> A, shared_ptr<Value> B) {
}

// ediv A \oslash B
shared_ptr<Value> fns::ediv(shared_ptr<Value> A, shared_ptr<Value> B) {
}

// relu A
shared_ptr<Value> fns::relu(shared_ptr<Value> A) {
}

// exp A
shared_ptr<Value> fns::exp(shared_ptr<Value> A) {
}

// log A
shared_ptr<Value> fns::log(shared_ptr<Value> A) {
}

// sum-reduce A (axis=k)
shared_ptr<Value> fns::sum_reduce(shared_ptr<Value> A, int axis, bool keepdims) {
}

// max-reduce A (axis=k)
shared_ptr<Value> fns::max_reduce(shared_ptr<Value> A, int axis, bool keepdims) {
}

// leaf
shared_ptr<Value> fns::leaf(Tensor result) {
}

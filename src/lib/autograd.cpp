#include "autograd.h"
using namespace autograd;

// ---- ctors ----

// creates new node which points to existing nodes; compute result in place.
// reduction functions MUST pass axis and keepdims - it's asserted.
Value::Value(FnType f_type, const vec<ValuePtr> &adj, int axis, bool keepdims)
    : f_type(f_type), adj(adj), axis(axis), keepdims(keepdims) {
    compute_result();
}

// ---- computation ----

// compute cached result, assuming adj results are all populated
void Value::compute_result() {
    switch (f_type) {
    case FnType::Add:
        result = adj[0]->result + adj[1]->result;
        break;
    case FnType::Ediv:
        result = adj[0]->result.ediv(adj[1]->result);
        break;
    case FnType::Exp:
        result = adj[0]->result.apply([](double x){return exp(x);});
        break;
    case FnType::Hadamard:
        result = adj[0]->result.hadamard(adj[1]->result);
        break;
    case FnType::Log:
        result = adj[0]->result.apply([](double x){return log(x);});
        break;
    case FnType::Matmul:
        result = adj[0]->result * adj[1]->result;
        break;
    case FnType::Relu:
        result = adj[0]->result.apply([](double x){return max(0., x);});
        break;
    case FnType::SumReduce:
        result = adj[0]->result.sum(axis, keepdims);
        break;
    case FnType::MaxReduce:
        // argmax mask via one-hot argmax odot with A, then reduce-sum to compress the axis
        result = adj[0]->result.argmax(axis).hadamard(adj[0]->result).sum(axis, keepdims);
        break;
    case FnType::Leaf:
        break;
    }
}

// implements g_{fn, i}: partial of root wrt positional arg i.
// axis field will be used if f_type is applicable.
static Tensor fn_g(FnType f_type, int i, const Tensor &G, const vec<ValuePtr> &args, int axis = -1) {
    switch (f_type) {
    case FnType::Matmul:
        if (i == 0) {
            Tensor &B = args[1]->result;
            return G * B.transpose();
        } else if (i == 1) {
            Tensor &A = args[0]->result;
            return A.transpose() * G;
        }
        break;
    case FnType::Add:
        return G;
    case FnType::Hadamard:
        return G.hadamard(args[1-i]->result);
    case FnType::Ediv: {
        Tensor &B = args[1]->result;
        if (i == 0) {
            return G.ediv(B);
        } else if (i == 1) {
            Tensor &A = args[0]->result;
            Tensor negG = G.apply([](double x){return -x;});
            Tensor bsq = B.apply([](double x){return x*x;});
            return negG.hadamard(A).hadamard(bsq);
        }
    } break;
    case FnType::Relu: {
        Tensor &A = args[0]->result;
        Tensor filter_A = A.apply([](double x){return x > 0 ? 1. : 0.;});
        return G.hadamard(filter_A);
    } break;
    case FnType::Exp: {
        Tensor &A = args[0]->result;
        Tensor exp_A = A.apply([](double x){return exp(x);});
        return G.hadamard(exp_A);
    } break;
    case FnType::Log: {
        Tensor &A = args[0]->result;
        Tensor inv_A = A.apply([](double x){return 1./x;});
        return G.hadamard(inv_A);
    } break;
    case FnType::SumReduce: {
        Tensor broadG = G;
        broadG.broadcast(args[0]->result.shape);
        return broadG;
    } break;
    case FnType::MaxReduce: {
        return G.argmax(axis);
    } break;
    case FnType::Leaf: {
    } break;
    }

    __builtin_unreachable();
}

// add chain rule contrib to grads of children in adj
void Value::add_child_grads() {
    for (int i = 0; i < sz(adj); ++i) {
        adj[i]->grad += fn_g(f_type, i, grad, adj, axis);
    }
}

// ---- public API ----

// traverse DAG topologically and compute grads
// root must be scalar. clears all reachable grads to 0 first.
void compute_all_grads(ValuePtr root) {
}

// matmul AB
ValuePtr fns::matmul(ValuePtr A, ValuePtr B) {
    return make_shared<Value>(
        FnType::Matmul,
        vec<ValuePtr>{A,B}
    );
}

// add A+B
ValuePtr fns::add(ValuePtr A, ValuePtr B) {
    return make_shared<Value>(
        FnType::Add,
        vec<ValuePtr>{A,B}
    );
}

// hadamard A \odot B
ValuePtr fns::hadamard(ValuePtr A, ValuePtr B) {
    return make_shared<Value>(
        FnType::Hadamard,
        vec<ValuePtr>{A,B}
    );
}

// ediv A \oslash B
ValuePtr fns::ediv(ValuePtr A, ValuePtr B) {
    return make_shared<Value>(
        FnType::Ediv,
        vec<ValuePtr>{A,B}
    );
}

// relu A
ValuePtr fns::relu(ValuePtr A) {
    return make_shared<Value>(
        FnType::Relu,
        vec<ValuePtr>{A}
    );
}

// exp A
ValuePtr fns::exp(ValuePtr A) {
    return make_shared<Value>(
        FnType::Exp,
        vec<ValuePtr>{A}
    );
}

// log A
ValuePtr fns::log(ValuePtr A) {
    return make_shared<Value>(
        FnType::Log,
        vec<ValuePtr>{A}
    );
}

// sum-reduce A (axis=k)
ValuePtr fns::sum_reduce(ValuePtr A, int axis, bool keepdims) {
    return make_shared<Value>(
        FnType::SumReduce,
        vec<ValuePtr>{A},
        axis, keepdims
    );
}

// max-reduce A (axis=k)
ValuePtr fns::max_reduce(ValuePtr A, int axis, bool keepdims) {
    return make_shared<Value>(
        FnType::MaxReduce,
        vec<ValuePtr>{A},
        axis, keepdims
    );
}

// leaf
ValuePtr fns::leaf(Tensor result) {
    ValuePtr leaf = make_shared<Value>(FnType::Leaf, vec<ValuePtr>{});
    leaf->result = result;
    return leaf;
}

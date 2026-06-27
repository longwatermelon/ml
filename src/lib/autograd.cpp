#include "autograd.h"
#include <unordered_set>
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
// axis & keepdims fields will be used if f_type is applicable.
static Tensor fn_g(FnType f_type, int i, const Tensor &G, const vec<ValuePtr> &args, int axis = -1, bool keepdims = true) {
    Tensor out;
    switch (f_type) {
    case FnType::Matmul:
        if (i == 0) {
            Tensor &B = args[1]->result;
            out = G * B.transpose();
        } else if (i == 1) {
            Tensor &A = args[0]->result;
            out = A.transpose() * G;
        }
        break;
    case FnType::Add:
        out = G;
        break;
    case FnType::Hadamard:
        out = G.hadamard(args[1-i]->result);
        break;
    case FnType::Ediv: {
        Tensor &B = args[1]->result;
        if (i == 0) {
            out = G.ediv(B);
        } else if (i == 1) {
            Tensor &A = args[0]->result;
            Tensor negG = G.apply([](double x){return -x;});
            Tensor bsq = B.apply([](double x){return x*x;});
            out = negG.hadamard(A).ediv(bsq);
        }
    } break;
    case FnType::Relu: {
        Tensor &A = args[0]->result;
        Tensor filter_A = A.apply([](double x){return x > 0 ? 1. : 0.;});
        out = G.hadamard(filter_A);
    } break;
    case FnType::Exp: {
        Tensor &A = args[0]->result;
        Tensor exp_A = A.apply([](double x){return exp(x);});
        out = G.hadamard(exp_A);
    } break;
    case FnType::Log: {
        Tensor &A = args[0]->result;
        Tensor inv_A = A.apply([](double x){return 1./x;});
        out = G.hadamard(inv_A);
    } break;
    case FnType::SumReduce: {
        vec<int> new_shape = args[0]->result.shape;
        new_shape[axis] = 1;

        Tensor broadG = G;
        if (!keepdims) broadG.reshape(new_shape);
        broadG.broadcast(args[0]->result.shape);
        out = broadG;
    } break;
    case FnType::MaxReduce: {
        vec<int> new_shape = args[0]->result.shape;
        new_shape[axis] = 1;

        out = G;
        if (!keepdims) out.reshape(new_shape);
        out.broadcast(args[0]->result.shape);
        out = out.hadamard(args[0]->result.argmax(axis));
    } break;
    case FnType::Leaf: {
    } break;
    }

    // collect gradients back to original shape, if broadcasted
    out.unbroadcast(args[i]->result.shape);
    return out;
}

// add chain rule contrib to grads of children in adj
void Value::add_child_grads() {
    for (int i = 0; i < sz(adj); ++i) {
        adj[i]->grad += fn_g(f_type, i, grad, adj, axis, keepdims);
    }
}

// ---- public API ----

// traverse DAG topologically and compute grads
// root must be scalar. clears all reachable grads to 0 first.
void compute_all_grads(ValuePtr root) {
    // traverse dfs post-order (eval children before parents), then reverse to get topological order.
    vec<ValuePtr> nodes_ord;
    unordered_set<Value*> seen;
    auto dfs = [&](ValuePtr u, auto &&self) -> void {
        if (seen.count(u.get()) > 0) return;
        seen.insert(u.get());

        for (ValuePtr child : u->adj) {
            self(child, self);
        }

        nodes_ord.push_back(u);
    };
    dfs(root, dfs);

    // zero all grads, except for root which should have ∂root/∂root = 1
    for (ValuePtr node : nodes_ord) {
        node->grad = node->result.apply([](double x){return 0.;});
    }
    root->grad = root->result.apply([](double x){return 1.;});

    // evaluate in topo order (backwards post-order)
    for (int i = sz(nodes_ord)-1; i >= 0; --i) {
        nodes_ord[i]->add_child_grads();
    }
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

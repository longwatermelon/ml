#include "autograd.h"
#include <unordered_set>
using namespace autograd;

// ---- ctors ----

// creates new node which points to existing nodes; compute result in place.
Value::Value(FnType f_type, const vec<shared_ptr<Value>> &adj)
    : f_type(f_type), adj(adj) {
    compute_result();
}

// for reduction nodes (max reduce, sum reduce).
Value::Value(FnType f_type, const vec<shared_ptr<Value>> &adj, int axis, bool keepdims)
    : f_type(f_type), adj(adj), axis(axis), keepdims(keepdims) {
    compute_result();
}

// for reshape nodes.
Value::Value(FnType f_type, const vec<shared_ptr<Value>> &adj, const vec<int> &new_shape)
    : f_type(f_type), adj(adj), new_shape(new_shape) {
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
    case FnType::Reshape:
        result = adj[0]->result;
        result.reshape(new_shape);
        break;
    case FnType::Leaf:
        break;
    }
}

// implements g_{fn, i}: partial of root wrt positional arg i.
// axis & keepdims fields will be used if f_type is applicable.
// given that it's X = f(args) where f is described by f_type, this calculates ∂J/∂args[i] = ∂J/∂X ∂X/∂args[i], evaluated at args.
// G = ∂J/∂X.
static Tensor fn_g(FnType f_type, int i, const Tensor &G, const vec<shared_ptr<Value>> &args, int axis, bool keepdims) {
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
    case FnType::Reshape: {
        out = G;
        out.reshape(args[0]->result.shape);
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

// ---- gtensor ctors ----

// init with a tensor
GTensor::GTensor(const Tensor &val) {
    value = make_shared<Value>(FnType::Leaf, vec<shared_ptr<Value>>{});
    value->result = val;
}

// tensor constructor
GTensor::GTensor(const vec<int> &shape, double value) {
    Tensor t(shape, value);
    this->value = make_shared<Value>(FnType::Leaf, vec<shared_ptr<Value>>{});
    this->value->result = t;
}

// tensor constructor
GTensor::GTensor(const vec<double> &data_1d) {
    Tensor t(data_1d);
    this->value = make_shared<Value>(FnType::Leaf, vec<shared_ptr<Value>>{});
    this->value->result = t;
}

// tensor constructor
GTensor::GTensor(const vec2<double> &data_2d) {
    Tensor t(data_2d);
    this->value = make_shared<Value>(FnType::Leaf, vec<shared_ptr<Value>>{});
    this->value->result = t;
}

// ---- gtensor operators ----

// matmul
GTensor GTensor::operator*(const GTensor &o) const {
    GTensor out;
    out.value = make_shared<Value>(
        FnType::Matmul,
        vec<shared_ptr<Value>>{this->value, o.value}
    );
    return out;
}

// add
GTensor GTensor::operator+(const GTensor &o) const {
    GTensor out;
    out.value = make_shared<Value>(
        FnType::Add,
        vec<shared_ptr<Value>>{this->value, o.value}
    );
    return out;
}

// element-wise mul
GTensor GTensor::hadamard(const GTensor &o) const {
    GTensor out;
    out.value = make_shared<Value>(
        FnType::Hadamard,
        vec<shared_ptr<Value>>{this->value, o.value}
    );
    return out;
}

// element-wise div
GTensor GTensor::ediv(const GTensor &o) const {
    GTensor out;
    out.value = make_shared<Value>(
        FnType::Ediv,
        vec<shared_ptr<Value>>{this->value, o.value}
    );
    return out;
}

// element-wise relu
GTensor GTensor::relu() const {
    GTensor out;
    out.value = make_shared<Value>(
        FnType::Relu,
        vec<shared_ptr<Value>>{this->value}
    );
    return out;
}

// element-wise exp
GTensor GTensor::exp() const {
    GTensor out;
    out.value = make_shared<Value>(
        FnType::Exp,
        vec<shared_ptr<Value>>{this->value}
    );
    return out;
}

// element-wise log
GTensor GTensor::log() const {
    GTensor out;
    out.value = make_shared<Value>(
        FnType::Log,
        vec<shared_ptr<Value>>{this->value}
    );
    return out;
}

// sum reduce
GTensor GTensor::sum_reduce(int axis, bool keepdims) const {
    GTensor out;
    out.value = make_shared<Value>(
        FnType::SumReduce,
        vec<shared_ptr<Value>>{this->value},
        axis, keepdims
    );
    return out;
}

// max reduce
GTensor GTensor::max_reduce(int axis, bool keepdims) const {
    GTensor out;
    out.value = make_shared<Value>(
        FnType::MaxReduce,
        vec<shared_ptr<Value>>{this->value},
        axis, keepdims
    );
    return out;
}

// negate
GTensor GTensor::operator-() const {
    GTensor neg1({1}, -1.);
    return neg1.hadamard(*this);
}

// subtract
GTensor GTensor::operator-(const GTensor &o) const {
    return *this + (-o);
}

// reshape
GTensor GTensor::reshape(const vec<int> &new_shape) const {
    GTensor out;
    out.value = make_shared<Value>(
        FnType::Reshape,
        vec<shared_ptr<Value>>{this->value},
        new_shape
    );
    return out;
}

// ---- autograd ----

// compute all grads: ∂this/∂reachable
// traverses DAG topologically. this must be scalar.
// clears all reachable grads to 0 first.
void GTensor::compute_all_grads() {
    shared_ptr<Value> root = value;
    assert(root->result.num_el() == 1);

    // traverse dfs post-order (eval children before parents), then reverse to get topological order.
    vec<shared_ptr<Value>> nodes_ord;
    unordered_set<Value*> seen;
    auto dfs = [&](shared_ptr<Value> u, auto &&self) -> void {
        if (seen.count(u.get()) > 0) return;
        seen.insert(u.get());

        for (shared_ptr<Value> child : u->adj) {
            self(child, self);
        }

        nodes_ord.push_back(u);
    };
    dfs(root, dfs);

    // zero all grads, except for root which should have ∂root/∂root = 1
    for (shared_ptr<Value> node : nodes_ord) {
        node->grad = node->result.apply([](double x){return 0.;});
    }
    root->grad = root->result.apply([](double x){return 1.;});

    // evaluate in topo order (backwards post-order)
    for (int i = sz(nodes_ord)-1; i >= 0; --i) {
        nodes_ord[i]->add_child_grads();
    }
}

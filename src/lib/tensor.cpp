#include "tensor.h"
#include <cassert>

// # elements given shape
static int numel(const vec<int> &shape) {
    int prod = 1;
    for (int i = 0; i < sz(shape); ++i) {
        prod *= shape[i];
    }
    return prod;
}

// shape -> stride, assume contiguous
static vec<int> shape2stride(const vec<int> &shape) {
    int prod = 1;
    vec<int> stride(sz(shape));
    for (int i = sz(shape)-1; i >= 0; --i) {
        stride[i] = prod;
        prod *= shape[i];
    }
    return stride;
}

// return if advance successful - false if can't advance anymore
static bool advance_ind(vec<int> &cur, const vec<int> &limits) {
    if (cur.empty()) {
        return false;
    }

    int ptr = sz(cur)-1;
    cur[ptr]++;
    while (cur[ptr] >= limits[ptr]) {
        if (ptr == 0) {
            return false;
        }

        cur[ptr] = 0;
        ptr--;
        cur[ptr]++;
    }

    return true;
}

// returns minimum possible sized shape that a,b can both broadcast to
static vec<int> parent_shape(const vec<int> &a, const vec<int> &b) {
    // pre-check: impossible to broadcast?
    for (int i = 0; i < min(sz(a), sz(b)); ++i) {
        int a_ind = sz(a)-1-i;
        int b_ind = sz(b)-1-i;
        assert(a[a_ind] == b[b_ind] || (a[a_ind] == 1 || b[b_ind] == 1));
    }

    // find parent shape
    vec<int> parent(max(sz(a), sz(b)));
    int aptr = sz(a)-1, bptr = sz(b)-1;
    for (int i = sz(parent)-1; i >= 0; --i) {
        parent[i] = max(
            aptr < 0 ? 1 : a[aptr],
            bptr < 0 ? 1 : b[bptr]
        );
        aptr--;
        bptr--;
    }
    return parent;
}

// ---- constructors ----

// 1d tensor from a vector
Tensor::Tensor(const vec<double> &data_1d) {
    shape = {sz(data_1d)};
    stride = {1};
    data = data_1d;
}

// 2d tensor from a 2d vector
Tensor::Tensor(const vec2<double> &data_2d) {
    shape = {sz(data_2d), sz(data_2d[0])};
    stride = {sz(data_2d[0]), 1};
    data.resize(shape[0] * shape[1]);
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            at({i,j}) = data_2d[i][j];
        }
    }
}

// tensor with given shape, filled with value
Tensor::Tensor(const vec<int> &shape, double value) {
    this->shape = shape;
    stride = shape2stride(shape);
    int n = numel(shape);
    data = vec<double>(n, value);
}

// ---- shape ops ----

// change shape without changing data
void Tensor::reshape(const vec<int> &new_shape) {
    assert(numel(shape) == numel(new_shape));
    *this = make_contiguous();
    shape = new_shape;
    stride = shape2stride(shape);
}

// broadcast dims of size 1 to match new_shape
void Tensor::broadcast(const vec<int> &new_shape) {
    pad_shape(new_shape);
    for (int i = 0; i < sz(shape); ++i) {
        if (shape[i] == 1) {
            stride[i] = 0;
            shape[i] = new_shape[i];
        } else {
            assert(shape[i] == new_shape[i]);
        }
    }
}

// sum-reduce along axes where target has size 1
void Tensor::unbroadcast(const vec<int> &target) {
}

// permute dimensions: new shape[i] = old shape[p[i]]
void Tensor::permute(const vec<int> &p) {
}

// left-pads shape, recomputes stride
void Tensor::pad_shape(const vec<int> &target) {
    vec<int> ones(sz(target)-sz(shape), 1);
    vec<int> strides(sz(target)-sz(shape), numel(shape));
    shape.insert(begin(shape), all(ones));
    stride.insert(begin(stride), all(strides));
}

// consolidate data, become contiguous again
Tensor Tensor::make_contiguous() const {
    int n = sz(shape);
    vec<int> cur(n);
    Tensor t(shape, 0.);
    do {
        t.at(cur) = at(cur);
    } while (advance_ind(cur, shape));
    return t;
}

// ---- element access ----

// mutable element access by multi-dim index
double &Tensor::at(const vec<int> &ind) {
    assert(sz(ind) == sz(shape));

    int flat_ind = 0;
    for (int i = 0; i < sz(ind); ++i) {
        flat_ind += stride[i] * ind[i];
    }

    assert(0 <= flat_ind && flat_ind < sz(data));
    return data[flat_ind];
}

// const element access by multi-dim index
double Tensor::at(const vec<int> &ind) const {
    assert(sz(ind) == sz(shape));

    int flat_ind = 0;
    for (int i = 0; i < sz(ind); ++i) {
        flat_ind += stride[i] * ind[i];
    }

    assert(0 <= flat_ind && flat_ind < sz(data));
    return data[flat_ind];
}

// ---- element-wise arithmetic ----

// element-wise sum
Tensor Tensor::operator+(const Tensor &o) const {
}

// element-wise difference
Tensor Tensor::operator-(const Tensor &o) const {
}

// in-place element-wise addition
Tensor &Tensor::operator+=(const Tensor &o) {
}

// in-place element-wise subtraction
Tensor &Tensor::operator-=(const Tensor &o) {
}

// element-wise product
Tensor Tensor::hadamard(const Tensor &o) const {
}

// element-wise division
Tensor Tensor::ediv(const Tensor &o) const {
}

// negate every element
Tensor Tensor::operator-() const {
    return apply([](double x) { return -x; });
}

// ---- matrix ops ----

// matmul on last two dims, batched over all leading dims
Tensor Tensor::operator*(const Tensor &o) const {
}

// transpose last two dims
Tensor Tensor::transpose() const {
}

// ---- element-wise function application ----

// return a new tensor with f applied to every element
Tensor Tensor::apply(const std::function<double(double)> &f) const {
    Tensor out = *this;
    vec<int> cur(sz(shape), 0);
    do {
        out.at(cur) = f(out.at(cur));
    } while (advance_ind(cur, shape));

    return out;
}

// applies function between two tensors, auto-broadcasts up to one tensor if needed
Tensor Tensor::apply(const Tensor &o, const std::function<double(double, double)> &f) const {
    Tensor out = *this, oth = o;

    // broadcast to match shapes
    vec<int> parent = parent_shape(out.shape, oth.shape);
    out.broadcast(parent);
    oth.broadcast(parent);

    // apply function, compute result
    Tensor result(parent, 0.);
    vec<int> cur(sz(parent), 0);
    do {
        result.at(cur) = f(out.at(cur), oth.at(cur));
    } while (advance_ind(cur, parent));

    return result;
}

// apply f to every element in place
Tensor &Tensor::apply_inplace(const std::function<double(double)> &f) {
}

// ---- reductions ----

// sum along an axis, shape[axis]=1 if keepdims, or axis is removed if not
Tensor Tensor::sum(int axis, bool keepdims) const {
    int n = sz(shape);

    // set up surrounding ind iteration (excluding axis)
    vec<int> cur(n-1, 0), limits = shape;
    limits.erase(begin(limits) + axis);

    // new summed tensor shape
    vec<int> new_shape = limits;
    if (keepdims) {
        new_shape.insert(begin(new_shape) + axis, 1);
    }
    Tensor t(new_shape, 0.);

    // iterate over all axis-exclude inds, flatten axis
    while (true) {
        // set up target (for t) and iter (for this)
        vec<int> target_pos = cur;
        vec<int> iter_pos = target_pos;
        iter_pos.insert(begin(iter_pos) + axis, 0);
        if (keepdims) {
            target_pos.insert(begin(target_pos) + axis, 0);
        }

        // iter & sum into t
        for (int i = 0; i < shape[axis]; ++i) {
            iter_pos[axis] = i;
            t.at(target_pos) += at(iter_pos);
        }

        // advance
        if (!advance_ind(cur, limits)) {
            break;
        }
    }

    return t;
}

// index of the minimum along an axis, result has shape[axis]=1
Tensor Tensor::argmin(int axis) const {
}

// index of the maximum along an axis, result has shape[axis]=1
Tensor Tensor::argmax(int axis) const {
}

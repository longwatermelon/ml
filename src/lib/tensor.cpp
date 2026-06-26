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

// ---- ctors ----

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

// reinterpret same data with different shape
void Tensor::reshape(const vec<int> &new_shape) {
    assert(numel(shape) == numel(new_shape));
    *this = materialize();
    shape = new_shape;
    stride = shape2stride(shape);
}

// broadcast shape to match new_shape
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

// sum-reduce along all axes i where shape[i]=1
void Tensor::unbroadcast(const vec<int> &target) {
    *this = materialize();
    assert(sz(target) <= sz(shape));

    // delete leading axes that broadcast padded in
    int extra = sz(shape) - sz(target);
    for (int i = 0; i < extra; ++i) {
        *this = sum(0, false);
    }

    // reduce axes in target which expanded from size 1
    for (int i = 0; i < sz(target); ++i) {
        assert(target[i] == shape[i] || target[i] == 1);
        if (target[i] == 1 && shape[i] > 1) {
            *this = sum(i, true);
        }
    }
}

// re-order axis arg order while maintaining semantic meaning of axes
void Tensor::permute(const vec<int> &p) {
    // check that p is valid & a permutation
    assert(sz(p) == sz(shape));
    int n = sz(p);
    vec<int> sorted = p;
    sort(all(sorted));
    for (int i = 0; i < n; ++i) {
        assert(sorted[i] == i);
    }

    // permute shape and stride
    vec<int> new_shape(n), new_stride(n);
    for (int i = 0; i < n; ++i) {
        new_shape[i] = shape[p[i]];
        new_stride[i] = stride[p[i]];
    }
    shape = new_shape;
    stride = new_stride;
}

// left-pad axes of current shape to match target shape's dimension cnt
void Tensor::pad_shape(const vec<int> &target) {
    assert(sz(target) >= sz(shape));
    vec<int> ones(sz(target)-sz(shape), 1);
    vec<int> strides(sz(target)-sz(shape), numel(shape));
    shape.insert(begin(shape), all(ones));
    stride.insert(begin(stride), all(strides));
}

// consolidate data, become contiguous again
Tensor Tensor::materialize() const {
    // skip materialize if already materialized
    if (is_contiguous()) {
        return *this;
    }

    // materialize
    int n = sz(shape);
    vec<int> cur(n);
    Tensor t(shape, 0.);
    do {
        t.at(cur) = at(cur);
    } while (advance_ind(cur, shape));
    return t;
}

// return if tensor is contiguous
bool Tensor::is_contiguous() const {
    return stride == shape2stride(shape) && sz(data) == numel(shape);
}

// ---- element access ----

// lvalue ref
double &Tensor::at(const vec<int> &ind) {
    assert(sz(ind) == sz(shape));

    int flat_ind = 0;
    for (int i = 0; i < sz(ind); ++i) {
        assert(0 <= ind[i] && ind[i] < shape[i]);
        flat_ind += stride[i] * ind[i];
    }

    assert(0 <= flat_ind && flat_ind < sz(data));
    return data[flat_ind];
}

// rvalue
double Tensor::at(const vec<int> &ind) const {
    assert(sz(ind) == sz(shape));

    int flat_ind = 0;
    for (int i = 0; i < sz(ind); ++i) {
        assert(0 <= ind[i] && ind[i] < shape[i]);
        flat_ind += stride[i] * ind[i];
    }

    assert(0 <= flat_ind && flat_ind < sz(data));
    return data[flat_ind];
}

// ---- arithmetic / operations ----

// element-wise sum
Tensor Tensor::operator+(const Tensor &o) const {
    return apply(o, [](double x, double y){return x+y;});
}

// element-wise difference
Tensor Tensor::operator-(const Tensor &o) const {
    return apply(o, [](double x, double y){return x-y;});
}

// in-place element-wise addition
Tensor &Tensor::operator+=(const Tensor &o) {
    apply_inplace(o, [](double x, double y){return x+y;});
    return *this;
}

// in-place element-wise subtraction
Tensor &Tensor::operator-=(const Tensor &o) {
    apply_inplace(o, [](double x, double y){return x-y;});
    return *this;
}

// element-wise prod
Tensor Tensor::hadamard(const Tensor &o) const {
    return apply(o, [](double x, double y){return x*y;});
}

// element-wise div
Tensor Tensor::ediv(const Tensor &o) const {
    return apply(o, [](double x, double y){return x/y;});
}

// unary negation
Tensor Tensor::operator-() const {
    return apply([](double x) { return -x; });
}

// matmul on least significant two axes, parallelized across the rest
Tensor Tensor::operator*(const Tensor &o) const {
    Tensor lhs = *this, rhs = o;

    // broadcasting
    assert(sz(lhs.shape) >= 2 && sz(rhs.shape) >= 2);
    vec<int> parent = parent_shape(lhs.shape, rhs.shape);
    lhs.broadcast(parent);
    rhs.broadcast(parent);
    int nd = sz(parent);

    // check shape: lhs is matrices [..., n, m] and rhs is matrices [..., m, k]
    assert(lhs.shape[nd-1] == rhs.shape[nd-2]);
    int n = lhs.shape[nd-2];
    int m = lhs.shape[nd-1];
    int k = rhs.shape[nd-1];

    // prep batch matmuls
    vec<int> batch_cur(n-2, 0);
    vec<int> batch_lim(begin(parent), end(parent)-2);

    // out tensor is batches of n*k matrices
    vec<int> out_shape = batch_lim;
    out_shape.push_back(n);
    out_shape.push_back(k);
    Tensor out(out_shape, 0.);

    // batch matmuls
    do {
        // out matrix ptr
        vec<int> cur = batch_cur;
        cur.push_back(0);
        cur.push_back(0);
        vec<int> lhs_cur = cur;
        vec<int> rhs_cur = cur;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                cur[nd-2] = i;
                cur[nd-1] = j;
                // vector dot: lhs row i, rhs col k
                for (int x = 0; x < m; ++x) {
                    lhs_cur[nd-2] = i;
                    lhs_cur[nd-1] = x;
                    rhs_cur[nd-2] = x;
                    rhs_cur[nd-1] = k;
                    out.at(cur) += lhs.at(lhs_cur) * rhs.at(rhs_cur);
                }
            }
        }
    } while (advance_ind(batch_cur, batch_lim));

    return out;
}

// transpose on least significant two axes, parallelized across the rest
Tensor Tensor::transpose() const {
}

// sum-reduce along an axis. keepdims = if reduced axis remains as len 1 or gets deleted.
Tensor Tensor::sum(int axis, bool keepdims) const {
    int n = sz(shape);
    assert(0 <= axis && axis < n);

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

// return tensor of argmin indices along axis arrays
Tensor Tensor::argmin(int axis) const {
}

// return tensor of argmax indices along axis arrays
Tensor Tensor::argmax(int axis) const {
}

// ---- functionals ----

// apply to copy of this
Tensor Tensor::apply(const std::function<double(double)> &f) const {
    Tensor out = *this;
    vec<int> cur(sz(shape), 0);
    out = out.materialize();
    do {
        out.at(cur) = f(out.at(cur));
    } while (advance_ind(cur, shape));

    return out;
}

// applies function between two tensors, auto-broadcasts both tensors as needed
Tensor Tensor::apply(const Tensor &o, const std::function<double(double, double)> &f) const {
    Tensor out = *this, oth = o;

    // broadcast to match shapes
    vec<int> parent = parent_shape(out.shape, oth.shape);
    out.broadcast(parent);
    oth.broadcast(parent);

    // apply function, compute result
    out = out.materialize();
    vec<int> cur(sz(parent), 0);
    do {
        out.at(cur) = f(out.at(cur), oth.at(cur));
    } while (advance_ind(cur, parent));

    return out;
}

// apply to this, return ref to this
Tensor &Tensor::apply_inplace(const std::function<double(double)> &f) {
    *this = materialize();
    vec<int> cur(sz(shape), 0);
    do {
        at(cur) = f(at(cur));
    } while (advance_ind(cur, shape));

    return *this;
}

// applies function between two tensors, store result in this, auto-broadcast both tensors as needed
Tensor &Tensor::apply_inplace(const Tensor &o, const std::function<double(double, double)> &f) {
    Tensor oth = o;

    // broadcast to match shapes
    vec<int> parent = parent_shape(shape, oth.shape);
    broadcast(parent);
    oth.broadcast(parent);

    // apply function
    *this = materialize();
    vec<int> cur(sz(parent), 0);
    do {
        at(cur) = f(at(cur), oth.at(cur));
    } while (advance_ind(cur, parent));

    return *this;
}

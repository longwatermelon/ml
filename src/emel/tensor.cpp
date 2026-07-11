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
Tensor Tensor::permute(const vec<int> &p) const {
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
    Tensor out(new_shape, 0.);
    out.stride = new_stride;
    out.data = data;
    return out;
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

    // must host matrices at minimum
    assert(sz(lhs.shape) >= 2 && sz(rhs.shape) >= 2);

    // check shape: lhs is matrices [..., n, m] and rhs is matrices [..., m, k]
    assert(lhs.shape[sz(lhs.shape)-1] == rhs.shape[sz(rhs.shape)-2]);
    int n = lhs.shape[sz(lhs.shape)-2];
    int m = lhs.shape[sz(lhs.shape)-1];
    int k = rhs.shape[sz(rhs.shape)-1];

    // materialize before broadcasting: guarantees each matrix is row-major
    // contiguous (row stride m/k, col stride 1) so the hot loop can run on raw
    // pointers.
    lhs = lhs.materialize();
    rhs = rhs.materialize();

    // isolate batch shapes to be broadcasted together; we don't want to
    // broadcast the matrix dimensions together
    vec<int> lhs_batch_shape = lhs.shape, rhs_batch_shape = rhs.shape;
    lhs_batch_shape.pop_back(); lhs_batch_shape.pop_back();
    rhs_batch_shape.pop_back(); rhs_batch_shape.pop_back();
    vec<int> batch_parent = parent_shape(lhs_batch_shape, rhs_batch_shape);

    // broadcast lhs, rhs against their batch parents
    // (matched matrix sizes, broadcast batch shapes)
    vec<int> lhs_batch_parent = batch_parent, rhs_batch_parent = batch_parent;
    lhs_batch_parent.push_back(n); lhs_batch_parent.push_back(m);
    rhs_batch_parent.push_back(m); rhs_batch_parent.push_back(k);
    lhs.broadcast(lhs_batch_parent);
    rhs.broadcast(rhs_batch_parent);
    int nd = sz(batch_parent)+2;

    // prep batch matmuls
    vec<int> batch_cur(nd-2, 0), batch_lim = batch_parent;

    // out tensor is batches of n*k matrices
    vec<int> out_shape = batch_lim;
    out_shape.push_back(n);
    out_shape.push_back(k);
    Tensor out(out_shape, 0.);

    // batch matmuls
    int batch_flat = 0; // out batch index
    do {
        // base offsets of this batch's matrices, via batch strides
        int lhs_base = 0, rhs_base = 0;
        for (int a = 0; a < nd-2; ++a) {
            lhs_base += batch_cur[a] * lhs.stride[a];
            rhs_base += batch_cur[a] * rhs.stride[a];
        }
        const double *A = lhs.data.data() + lhs_base;   // n*m, row-major
        const double *B = rhs.data.data() + rhs_base;   // m*k, row-major
        double *C = out.data.data() + batch_flat * n*k; // n*k, row-major

        // carry out matmul
        for (int i = 0; i < n; ++i) {
            double *Crow = C + i*k;
            for (int x = 0; x < m; ++x) {
                double a = A[i*m + x];
                const double *Brow = B + x*k;
                for (int j = 0; j < k; ++j) {
                    Crow[j] += a * Brow[j];
                }
            }
        }

        batch_flat++;
    } while (advance_ind(batch_cur, batch_lim));

    return out;
}

// transpose on least significant two axes, parallelized across the rest
Tensor Tensor::transpose() const {
    int n = sz(shape);
    vec<int> p(n);
    for (int i = 0; i < n; ++i) {
        p[i] = i;
    }
    swap(p[n-2], p[n-1]);
    return permute(p);
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

// max-reduce along an axis. keepdims = if reduced axis remains as len 1 or gets deleted.
Tensor Tensor::max(int axis, bool keepdims) const {
    int n = sz(shape);
    assert(0 <= axis && axis < n);

    // set up surrounding ind iteration (excluding axis)
    vec<int> cur(n-1, 0), limits = shape;
    limits.erase(begin(limits) + axis);

    // new maxed tensor shape
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

        // iter & max into t
        t.at(target_pos) = at(iter_pos);
        for (int i = 1; i < shape[axis]; ++i) {
            iter_pos[axis] = i;
            t.at(target_pos) = std::max(t.at(target_pos), at(iter_pos));
        }

        // advance
        if (!advance_ind(cur, limits)) {
            break;
        }
    }

    return t;
}

// return tensor of one-hot encoded argmaxes along axis arrays
Tensor Tensor::argmax(int axis) const {
    int n = sz(shape);
    assert(0 <= axis && axis < n);

    // iter over all other axes
    vec<int> parent_cur(n-1, 0);
    vec<int> parent_lim = shape;
    parent_lim.erase(begin(parent_lim) + axis);
    Tensor out(shape, 0.);
    do {
        // identify argmax
        vec<int> cur = parent_cur;
        cur.insert(begin(cur) + axis, 0);
        vec<int> mx_ind = cur;
        for (int i = 1; i < shape[axis]; ++i) {
            cur[axis] = i;
            if (at(cur) > at(mx_ind)) {
                mx_ind = cur;
            }
        }

        // set 1 in out at argmax
        out.at(mx_ind) = 1.;
    } while (advance_ind(parent_cur, parent_lim));

    return out;
}

// replace self with index mapping: new[ind] = this[I[ind]]. Requires I.shape = output shape + {rank(this.shape)}.
Tensor Tensor::gather(const Tensor &I) const {
    // shape assertion
    assert(I.shape.back() == sz(shape));

    // iter over entries in I
    vec<int> output_shape = I.shape;
    output_shape.pop_back();
    Tensor out(output_shape, 0.);
    vec<int> cur(sz(output_shape), 0);
    vec<int> lim = output_shape;
    do {
        // construct index
        vec<int> ind;
        cur.push_back(0);
        for (int i = 0; i < I.shape.back(); ++i) {
            cur.back() = i;
            ind.push_back((int)I.at(cur));
        }
        cur.pop_back();

        // assign
        out.at(cur) = at(ind);
    } while (advance_ind(cur, lim));

    return out;
}

// gather, except if this is 1D, we exclude the redundant trailing axis of length 1.
Tensor Tensor::gather_flat(const Tensor &I) const {
    assert(sz(shape) == 1);
    Tensor Ip = I;
    vec<int> new_shape = Ip.shape;
    new_shape.push_back(1);
    Ip.reshape(new_shape);
    return gather(Ip);
}

// softmax, composed of primitives
Tensor Tensor::softmax(int axis) const {
    Tensor out = *this;
    // nuemrical stability: subtract max logits
    out = out - out.max(axis, true);
    // exp all logits
    out = out.apply([](double x){return exp(x);});
    // denominators across axis
    Tensor denom = out.sum(axis, true);
    // divide to get probs
    out = out.ediv(denom);

    return out;
}

// ---- functionals ----

// apply to copy of this
Tensor Tensor::apply(const std::function<double(double)> &f) const {
    Tensor out = materialize();

    // apply function
    vec<double> &data = out.data;
    for (int i = 0; i < sz(data); ++i) {
        data[i] = f(data[i]);
    }

    return out;
}

// applies function between two tensors, auto-broadcasts both tensors as needed
Tensor Tensor::apply(const Tensor &o, const std::function<double(double, double)> &f) const {
    Tensor out = *this, oth = o;

    // check if one's shape is the suffix of the other
    auto suffix_match_apply = [&f](Tensor &lhs, Tensor &rhs, bool left_is_suffix) {
        Tensor &big = left_is_suffix ? rhs : lhs;
        Tensor &suffix = left_is_suffix ? lhs : rhs;

        if (lhs.is_contiguous() && rhs.is_contiguous() &&
                   sz(big.shape) > sz(suffix.shape)) {
            // rhs matches suffix of lhs's shape?
            bool match = true;
            int offset = sz(big.shape) - sz(suffix.shape);
            for (int i = 0; i < sz(suffix.shape); ++i) {
                if (big.shape[offset + i] != suffix.shape[i]) {
                    match = false;
                    break;
                }
            }

            // matches, run periodic read
            if (match) {
                int period = suffix.num_el();
                int big_numel = big.num_el();
                for (int i = 0; i < big_numel; ++i) {
                    if (left_is_suffix) {
                        big.data[i] = f(suffix.data[i % period], big.data[i]);
                    } else {
                        big.data[i] = f(big.data[i], suffix.data[i % period]);
                    }
                }
                return true;
            }
        }

        return false;
    };

    if (suffix_match_apply(out, oth, true)) {
        return oth;
    }
    if (suffix_match_apply(out, oth, false)) {
        return out;
    }

    // broadcast to match shapes
    vec<int> parent = parent_shape(out.shape, oth.shape);
    out.broadcast(parent);
    oth.broadcast(parent);

    // apply function: both contiguous; optimized raw indexing
    if (out.is_contiguous() && oth.is_contiguous()) {
        vec<double> &data_out = out.data;
        vec<double> &data_oth = oth.data;
        for (int i = 0; i < sz(data_out); ++i) {
            data_out[i] = f(data_out[i], data_oth[i]);
        }

        return out;
    }

    // apply function: non-special case, just do it unoptimized
    out = out.materialize();
    vec<int> cur(sz(out.shape), 0);
    vec<int> lim = out.shape;
    do {
        out.at(cur) = f(out.at(cur), oth.at(cur));
    } while (advance_ind(cur, lim));
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

// ---- getters ----

// # elements that exist in the tensor
int Tensor::num_el() const {
    return numel(shape);
}

// ---- save/load ----

// serialize to bytes
vec<uint8_t> Tensor::serialize() const {
    vec<uint8_t> bytes;

    // shape info
    Tensor out = materialize();
    append_bytes(bytes, (uint32_t)sz(out.shape));
    for (int i = 0; i < sz(shape); ++i) {
        append_bytes(bytes, (uint32_t)out.shape[i]);
    }

    // data info
    append_bytes(bytes, (uint64_t)out.data.size());
    append_bytes_count(bytes, out.data.data(), out.data.size() * sizeof(double));

    return bytes;
}

// deserialize from bytes
Tensor Tensor::deserialize(const vec<uint8_t> &bytes) {
    size_t pos = 0;
    return deserialize(bytes, pos);
}

// deserialize one tensor from bytes starting at pos, advancing pos past it
Tensor Tensor::deserialize(const vec<uint8_t> &bytes, size_t &pos) {
    Tensor out;

    // shape info
    uint32_t rank = read_bytes<uint32_t>(bytes, pos);
    out.shape.resize(rank);
    for (uint32_t i = 0; i < rank; ++i) {
        out.shape[i] = read_bytes<uint32_t>(bytes, pos);
    }

    // data info
    uint64_t data_len = read_bytes<uint64_t>(bytes, pos);
    out.data.resize(data_len);
    read_bytes_count<double>(bytes, pos, out.data.data(), data_len * sizeof(double));

    // recompute stride
    out.stride = shape2stride(out.shape);

    return out;
}


#include "tensor.h"
#include <cassert>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

// matmul C = A*B where A is n*m, B is m*k, C is n*k. C must be zeroed.
static void sgemm(const float *A, const float *B, float *C, int n, int m, int k) {
    if (n == 0 || m == 0 || k == 0) return;
#ifdef __APPLE__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, k, m, 1.f, A, m, B, k, 0.f, C, k);
#else
    // portable cpu fallback
    for (int i = 0; i < n; ++i) {
        float *Crow = C + i*k;
        for (int x = 0; x < m; ++x) {
            float a = A[i*m + x];
            const float *Brow = B + x*k;
            for (int j = 0; j < k; ++j) {
                Crow[j] += a * Brow[j];
            }
        }
    }
#endif
}

// split a contiguous shape around an axis: [outer, len, inner] so that
// flat index = (a*len + i)*inner + j for outer index a, axis index i, inner index j
static void axis_split(const vec<int> &shape, int axis, int &outer, int &len, int &inner) {
    outer = 1; inner = 1;
    for (int i = 0; i < axis; ++i) outer *= shape[i];
    for (int i = axis+1; i < sz(shape); ++i) inner *= shape[i];
    len = shape[axis];
}

// ---- ctors ----

// 1d tensor from a vector
Tensor::Tensor(const vec<float> &data_1d) {
    shape = {sz(data_1d)};
    stride = {1};
    data = data_1d;
}

// 2d tensor from a 2d vector
Tensor::Tensor(const vec2<float> &data_2d) {
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
Tensor::Tensor(const vec<int> &shape, float value) {
    this->shape = shape;
    stride = shape2stride(shape);
    int n = numel(shape);
    data = vec<float>(n, value);
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
    Tensor out(new_shape, 0.f);
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
    // skip this function if already materialized
    if (is_contiguous()) {
        return *this;
    }

    // materialize
    int n = sz(shape);
    vec<int> cur(n);
    Tensor t(shape, 0.f);
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
float &Tensor::at(const vec<int> &ind) {
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
float Tensor::at(const vec<int> &ind) const {
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
    return apply(o, [](float x, float y){return x+y;});
}

// element-wise difference
Tensor Tensor::operator-(const Tensor &o) const {
    return apply(o, [](float x, float y){return x-y;});
}

// in-place element-wise addition
Tensor &Tensor::operator+=(const Tensor &o) {
    apply_inplace(o, [](float x, float y){return x+y;});
    return *this;
}

// in-place element-wise subtraction
Tensor &Tensor::operator-=(const Tensor &o) {
    apply_inplace(o, [](float x, float y){return x-y;});
    return *this;
}

// element-wise prod
Tensor Tensor::hadamard(const Tensor &o) const {
    return apply(o, [](float x, float y){return x*y;});
}

// element-wise div
Tensor Tensor::ediv(const Tensor &o) const {
    return apply(o, [](float x, float y){return x/y;});
}

// unary negation
Tensor Tensor::operator-() const {
    return apply([](float x) { return -x; });
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

    // ensure matrices are contiguous, so hot loops can run on raw pointers
    lhs = lhs.materialize();
    rhs = rhs.materialize();

    // special fast path: 2d rhs means every batch matrix multiplies the same rhs, so
    // just stack all batch matrices sequentially vertically, all the rows in it
    // process in parallel. output shape [batch*n, m]
    if (sz(rhs.shape) == 2) {
        vec<int> out_shape = lhs.shape;
        out_shape.back() = k;
        Tensor out(out_shape, 0.f);
        vec<int> row_shape = lhs.shape;
        row_shape.pop_back();
        sgemm(lhs.data.data(), rhs.data.data(), out.data.data(),
              numel(row_shape), m, k);
        return out;
    }

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
    Tensor out(out_shape, 0.f);

    // batch matmuls
    int batch_flat = 0; // out batch index
    do {
        // base offsets of this batch's matrices, via batch strides
        int lhs_base = 0, rhs_base = 0;
        for (int a = 0; a < nd-2; ++a) {
            lhs_base += batch_cur[a] * lhs.stride[a];
            rhs_base += batch_cur[a] * rhs.stride[a];
        }
        const float *A = lhs.data.data() + lhs_base;   // n*m, row-major
        const float *B = rhs.data.data() + rhs_base;   // m*k, row-major
        float *C = out.data.data() + batch_flat * n*k; // n*k, row-major

        // carry out matmul
        sgemm(A, B, C, n, m, k);

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

    // fast path: contiguous data reduces with flat loops
    if (is_contiguous()) {
        vec<int> new_shape = shape;
        if (keepdims) new_shape[axis] = 1;
        else new_shape.erase(begin(new_shape) + axis);

        int outer, len, inner;
        axis_split(shape, axis, outer, len, inner);

        Tensor t(new_shape, 0.f);
        const float *src = data.data();
        float *dst = t.data.data();
        for (int a = 0; a < outer; ++a) {
            const float *block = src + a*len*inner;
            float *out = dst + a*inner;
            for (int i = 0; i < len; ++i) {
                const float *row = block + i*inner;
                for (int j = 0; j < inner; ++j) {
                    out[j] += row[j];
                }
            }
        }
        return t;
    }

    // set up surrounding ind iteration (excluding axis)
    vec<int> cur(n-1, 0), limits = shape;
    limits.erase(begin(limits) + axis);

    // new summed tensor shape
    vec<int> new_shape = limits;
    if (keepdims) {
        new_shape.insert(begin(new_shape) + axis, 1);
    }
    Tensor t(new_shape, 0.f);

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

    // special fast path: raw flat ptr iter for contiguous data
    if (is_contiguous()) {
        vec<int> new_shape = shape;
        if (keepdims) new_shape[axis] = 1;
        else new_shape.erase(begin(new_shape) + axis);

        int outer, len, inner;
        axis_split(shape, axis, outer, len, inner);

        Tensor t(new_shape, 0.f);
        const float *src = data.data();
        float *dst = t.data.data();
        for (int a = 0; a < outer; ++a) {
            const float *block = src + a*len*inner;
            float *out = dst + a*inner;
            for (int j = 0; j < inner; ++j) {
                out[j] = block[j];
            }
            for (int i = 1; i < len; ++i) {
                const float *row = block + i*inner;
                for (int j = 0; j < inner; ++j) {
                    out[j] = std::max(out[j], row[j]);
                }
            }
        }
        return t;
    }

    // set up surrounding ind iteration (excluding axis)
    vec<int> cur(n-1, 0), limits = shape;
    limits.erase(begin(limits) + axis);

    // new maxed tensor shape
    vec<int> new_shape = limits;
    if (keepdims) {
        new_shape.insert(begin(new_shape) + axis, 1);
    }
    Tensor t(new_shape, 0.f);

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

    // special fast path: raw flat ptr iter for contiguous data
    if (is_contiguous()) {
        int outer, len, inner;
        axis_split(shape, axis, outer, len, inner);

        Tensor out(shape, 0.f);
        const float *src = data.data();
        float *dst = out.data.data();
        for (int a = 0; a < outer; ++a) {
            const float *block = src + a*len*inner;
            for (int j = 0; j < inner; ++j) {
                int best = 0;
                float best_val = block[j];
                for (int i = 1; i < len; ++i) {
                    float v = block[i*inner + j];
                    if (v > best_val) {
                        best_val = v;
                        best = i;
                    }
                }
                dst[(a*len + best)*inner + j] = 1.f;
            }
        }
        return out;
    }

    // iter over all other axes
    vec<int> parent_cur(n-1, 0);
    vec<int> parent_lim = shape;
    parent_lim.erase(begin(parent_lim) + axis);
    Tensor out(shape, 0.f);
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
        out.at(mx_ind) = 1.f;
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
    Tensor out(output_shape, 0.f);
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

// gather, except if this is 1D, we exclude the redundant trailing axis of length 1
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
    out = out.apply([](float x){return exp(x);});
    // denominators across axis
    Tensor denom = out.sum(axis, true);
    // divide to get probs
    out = out.ediv(denom);

    return out;
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
    append_bytes_count(bytes, out.data.data(), out.data.size() * sizeof(float));

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
    read_bytes_count<float>(bytes, pos, out.data.data(), data_len * sizeof(float));

    // recompute stride
    out.stride = shape2stride(out.shape);

    return out;
}

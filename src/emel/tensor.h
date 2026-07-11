#pragma once
#include "util.h"
#include <cassert>
#include <utility>

/* 
 * General policy regarding tensors: before any operations that modify the tensor itself, first materialize it.
 * This avoids annoyances with accidentally modifying multiple entries in the non-contiguous tensor when we're
 * only trying to modify one.
*/

// ---- shape helpers ----

// # elements given shape
inline int numel(const vec<int> &shape) {
    int prod = 1;
    for (int i = 0; i < sz(shape); ++i) {
        prod *= shape[i];
    }
    return prod;
}

// shape -> stride, assume contiguous
inline vec<int> shape2stride(const vec<int> &shape) {
    int prod = 1;
    vec<int> stride(sz(shape));
    for (int i = sz(shape)-1; i >= 0; --i) {
        stride[i] = prod;
        prod *= shape[i];
    }
    return stride;
}

// returns minimum possible sized shape that a,b can both broadcast to
inline vec<int> parent_shape(const vec<int> &a, const vec<int> &b) {
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

struct Tensor {
    // think of stride[i] denoting the size taken up in the data vector by a sub-shape below it.
    // stride is how we index by sub-shapes as we go left in axes instead of individual entries;
    // individual entries are only sub-shapes for the least significant axis (the rightmost one).
    vec<int> shape, stride;
    vec<float> data;

    // ---- ctors ----

    Tensor() = default;

    Tensor(const vec<float> &data_1d);
    Tensor(const vec2<float> &data_2d);
    Tensor(const vec<int> &shape, float value);

    // ---- shape ops ----

    // reinterpret same data with different shape
    void reshape(const vec<int> &new_shape);
    // broadcast shape to match new_shape
    void broadcast(const vec<int> &new_shape);
    // sum-reduce along all axes i where shape[i]=1
    void unbroadcast(const vec<int> &shape);
    // re-order axis arg order while maintaining semantic meaning of axes
    Tensor permute(const vec<int> &p) const;
    // left-pad axes of current shape to match target shape's dimension cnt
    void pad_shape(const vec<int> &target);
    // consolidate data, become contiguous again
    Tensor materialize() const;
    // return if tensor is contiguous
    bool is_contiguous() const;

    // ---- element access ----

    // lvalue ref
    float &at(const vec<int> &ind);
    // rvalue
    float at(const vec<int> &ind) const;

    // ---- arithmetic / operations ----

    Tensor operator+(const Tensor &o) const;
    Tensor operator-(const Tensor &o) const;
    Tensor &operator+=(const Tensor &o);
    Tensor &operator-=(const Tensor &o);
    // element-wise prod
    Tensor hadamard(const Tensor &o) const;
    // element-wise div
    Tensor ediv(const Tensor &o) const;
    // unary negation
    Tensor operator-() const;
    // matmul on least significant two axes, parallelized across the rest
    Tensor operator*(const Tensor &o) const;
    // transpose on least significant two axes, parallelized across the rest
    Tensor transpose() const;
    // sum-reduce along an axis. keepdims = if reduced axis remains as len 1 or gets deleted.
    Tensor sum(int axis, bool keepdims) const;
    // max-reduce along an axis. keepdims = if reduced axis remains as len 1 or gets deleted.
    Tensor max(int axis, bool keepdims) const;
    // return tensor of one-hot encoded argmaxes along axis arrays
    Tensor argmax(int axis) const;
    // replace self with index mapping: new[ind] = this[I[ind]]. Requires I.shape = output shape + rank(this.shape).
    Tensor gather(const Tensor &I) const;
    // gather, except if this is 1D, we exclude the redundant trailing axis of length 1
    Tensor gather_flat(const Tensor &I) const;
    // softmax across specified axis
    Tensor softmax(int axis) const;

    // ---- functionals ----
    // templated (rather than std::function) so lambdas inline into the loops

    // apply to copy of this
    template <typename F>
    Tensor apply(const F &f) const {
        Tensor out = *this;
        out.apply_inplace(f);
        return out;
    }

    // applies function between two tensors, auto-broadcasts both tensors as needed
    template <typename F>
    Tensor apply(const Tensor &o, const F &f) const {
        Tensor out = *this;
        out.apply_inplace(o, f);
        return out;
    }

    // apply to this, return ref to this
    template <typename F>
    Tensor &apply_inplace(const F &f) {
        // must materialize before directly operating on logical entries
        if (!is_contiguous()) {
            *this = materialize();
        }

        for (float &value : data) {
            value = f(value);
        }

        return *this;
    }

    // applies function between two tensors, store result in this, auto-broadcast both tensors as needed
    template <typename F>
    Tensor &apply_inplace(const Tensor &o, const F &f) {
        // avoid copies and indexed access for the common equal-shape case
        if (shape == o.shape && is_contiguous() && o.is_contiguous()) {
            for (int i = 0; i < sz(data); ++i) {
                data[i] = f(data[i], o.data[i]);
            }
            return *this;
        }

        // scalar, exact-suffix, and row-broadcast optimizations
        if (is_contiguous() && o.is_contiguous()) {
            // rhs is scalar
            if (o.num_el() == 1) {
                vec<int> parent = parent_shape(shape, o.shape);
                float scalar = o.data[0];
                for (float &value : data) {
                    value = f(value, scalar);
                }
                shape = parent;
                stride = shape2stride(parent);
                return *this;
            }

            // lhs is scalar
            if (num_el() == 1) {
                vec<int> parent = parent_shape(shape, o.shape);
                float scalar = data[0];
                Tensor expanded(parent, 0.f);
                for (int i = 0; i < sz(expanded.data); ++i) {
                    expanded.data[i] = f(scalar, o.data[i]);
                }
                *this = std::move(expanded);
                return *this;
            }

            auto is_suffix = [](const vec<int> &big, const vec<int> &suffix) {
                if (sz(big) <= sz(suffix)) return false;

                int offset = sz(big) - sz(suffix);
                for (int i = 0; i < sz(suffix); ++i) {
                    if (big[offset + i] != suffix[i]) return false;
                }
                return true;
            };

            // lhs is suffix of rhs
            if (is_suffix(shape, o.shape)) {
                int period = o.num_el();
                for (int base = 0; base < sz(data); base += period) {
                    for (int i = 0; i < period; ++i) {
                        data[base + i] = f(data[base + i], o.data[i]);
                    }
                }
                return *this;
            }

            // rhs is suffix of lhs
            if (is_suffix(o.shape, shape)) {
                int period = num_el();
                Tensor expanded(o.shape, 0.f);
                for (int base = 0; base < sz(expanded.data); base += period) {
                    for (int i = 0; i < period; ++i) {
                        expanded.data[base + i] = f(data[i], o.data[base + i]);
                    }
                }
                *this = std::move(expanded);
                return *this;
            }

            // shapes match except small's last axis is 1 (per-row scalars,
            // e.g. softmax denominators / layernorm means)
            auto is_row_scalar = [](const vec<int> &big, const vec<int> &small) {
                if (sz(big) == 0 || sz(big) != sz(small)) return false;
                for (int i = 0; i + 1 < sz(big); ++i) {
                    if (big[i] != small[i]) return false;
                }
                return small.back() == 1;
            };

            // rhs holds per-row scalars for lhs's last axis
            if (is_row_scalar(shape, o.shape)) {
                int len = shape.back();
                int rows = num_el() / len;
                for (int r = 0; r < rows; ++r) {
                    float *row = data.data() + r*len;
                    float s = o.data[r];
                    for (int i = 0; i < len; ++i) {
                        row[i] = f(row[i], s);
                    }
                }
                return *this;
            }

            // lhs holds per-row scalars for rhs's last axis
            if (is_row_scalar(o.shape, shape)) {
                int len = o.shape.back();
                int rows = o.num_el() / len;
                Tensor expanded(o.shape, 0.f);
                for (int r = 0; r < rows; ++r) {
                    const float *row = o.data.data() + r*len;
                    float *out = expanded.data.data() + r*len;
                    float s = data[r];
                    for (int i = 0; i < len; ++i) {
                        out[i] = f(s, row[i]);
                    }
                }
                *this = std::move(expanded);
                return *this;
            }
        }

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
    int num_el() const;

    // ---- save/load ----

    // serialize to bytes
    vec<uint8_t> serialize() const;
    // deserialize from bytes
    static Tensor deserialize(const vec<uint8_t> &bytes);
    // deserialize one tensor from bytes starting at pos, advancing pos past it
    static Tensor deserialize(const vec<uint8_t> &bytes, size_t &pos);
};

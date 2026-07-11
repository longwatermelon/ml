#include <doctest/doctest.h>
#include "emel/tensor.h"

// check a tensor's logical shape and values
static void check_tensor(
    const Tensor &actual,
    const vec<int> &expected_shape,
    const vec<float> &expected_data
) {
    REQUIRE(actual.shape == expected_shape);
    REQUIRE(actual.num_el() == sz(expected_data));

    Tensor flat = actual.materialize();
    for (int i = 0; i < sz(expected_data); ++i) {
        CAPTURE(i);
        CHECK(flat.data[i] == doctest::Approx(expected_data[i]));
    }
}

TEST_SUITE_BEGIN("tensor");

TEST_CASE("tensor construction and element access") {
    Tensor one_dimensional(vec<float>{1.f, 2.f, 3.f});
    check_tensor(one_dimensional, {3}, {1.f, 2.f, 3.f});

    Tensor two_dimensional(vec2<float>{{1.f, 2.f}, {3.f, 4.f}});
    check_tensor(two_dimensional, {2, 2}, {1.f, 2.f, 3.f, 4.f});

    Tensor filled(vec<int>{2, 3}, 7.f);
    check_tensor(filled, {2, 3}, {7.f, 7.f, 7.f, 7.f, 7.f, 7.f});

    filled.at({1, 2}) = 9.f;
    CHECK(filled.at({1, 2}) == 9.f);
    CHECK(filled.is_contiguous());
}

TEST_CASE("tensor shape operations") {
    Tensor reshaped(vec<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    reshaped.reshape({2, 3});
    check_tensor(reshaped, {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

    Tensor broadcasted(vec<float>{1.f, 2.f, 3.f});
    broadcasted.broadcast({2, 3});
    check_tensor(broadcasted, {2, 3}, {1.f, 2.f, 3.f, 1.f, 2.f, 3.f});
    CHECK_FALSE(broadcasted.is_contiguous());
    CHECK(broadcasted.materialize().is_contiguous());

    broadcasted.unbroadcast({3});
    check_tensor(broadcasted, {3}, {2.f, 4.f, 6.f});

    Tensor axes(vec<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    axes.reshape({1, 2, 3});
    check_tensor(axes.permute({2, 1, 0}), {3, 2, 1}, {1.f, 4.f, 2.f, 5.f, 3.f, 6.f});
    check_tensor(axes.transpose(), {1, 3, 2}, {1.f, 4.f, 2.f, 5.f, 3.f, 6.f});
}

TEST_CASE("tensor arithmetic and functionals") {
    Tensor matrix(vec2<float>{{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});
    Tensor row(vec<float>{1.f, 2.f, 3.f});

    check_tensor(matrix + row, {2, 3}, {2.f, 4.f, 6.f, 5.f, 7.f, 9.f});
    check_tensor(matrix - row, {2, 3}, {0.f, 0.f, 0.f, 3.f, 3.f, 3.f});
    check_tensor(matrix.hadamard(row), {2, 3}, {1.f, 4.f, 9.f, 4.f, 10.f, 18.f});
    check_tensor(matrix.ediv(row), {2, 3}, {1.f, 1.f, 1.f, 4.f, 2.5f, 2.f});
    check_tensor(-row, {3}, {-1.f, -2.f, -3.f});

    Tensor inplace = matrix;
    inplace += row;
    inplace -= row;
    check_tensor(inplace, {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

    Tensor expanding_inplace = row;
    expanding_inplace -= matrix;
    check_tensor(expanding_inplace, {2, 3}, {0.f, 0.f, 0.f, -3.f, -3.f, -3.f});

    Tensor equal_shape = matrix;
    equal_shape += Tensor(vec2<float>{{1.f, 1.f, 1.f}, {2.f, 2.f, 2.f}});
    check_tensor(equal_shape, {2, 3}, {2.f, 3.f, 4.f, 6.f, 7.f, 8.f});

    Tensor scalar_rhs = matrix;
    scalar_rhs += Tensor({1}, 2.f);
    check_tensor(scalar_rhs, {2, 3}, {3.f, 4.f, 5.f, 6.f, 7.f, 8.f});

    Tensor scalar_lhs({1}, 2.f);
    scalar_lhs -= matrix;
    check_tensor(scalar_lhs, {2, 3}, {1.f, 0.f, -1.f, -2.f, -3.f, -4.f});

    Tensor leading_scalar_rhs = matrix;
    leading_scalar_rhs += Tensor({1, 1, 1}, 2.f);
    check_tensor(leading_scalar_rhs, {1, 2, 3}, {3.f, 4.f, 5.f, 6.f, 7.f, 8.f});

    check_tensor(row.apply([](float value) { return value * value; }), {3}, {1.f, 4.f, 9.f});
    row.apply_inplace([](float value) { return value + 0.5f; });
    check_tensor(row, {3}, {1.5f, 2.5f, 3.5f});

    Tensor transposed = matrix.transpose();
    transposed.apply_inplace([](float value) { return value * 2.f; });
    check_tensor(transposed, {3, 2}, {2.f, 8.f, 4.f, 10.f, 6.f, 12.f});
}

TEST_CASE("tensor matrix multiplication") {
    Tensor left(vec2<float>{{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});
    Tensor right(vec2<float>{{7.f, 8.f}, {9.f, 10.f}, {11.f, 12.f}});

    check_tensor(left * right, {2, 2}, {58.f, 64.f, 139.f, 154.f});
}

TEST_CASE("tensor reductions") {
    Tensor values(vec2<float>{{1.f, 5.f, 3.f}, {4.f, 2.f, 6.f}});

    check_tensor(values.sum(0, false), {3}, {5.f, 7.f, 9.f});
    check_tensor(values.sum(1, true), {2, 1}, {9.f, 12.f});
    check_tensor(values.max(0, true), {1, 3}, {4.f, 5.f, 6.f});
    check_tensor(values.max(1, false), {2}, {5.f, 6.f});
    check_tensor(values.argmax(1), {2, 3}, {0.f, 1.f, 0.f, 0.f, 0.f, 1.f});
}

TEST_CASE("tensor gathering") {
    Tensor values(vec2<float>{{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});
    Tensor indices(vec2<float>{{0.f, 1.f}, {1.f, 2.f}});
    check_tensor(values.gather(indices), {2}, {2.f, 6.f});

    Tensor flat(vec<float>{10.f, 20.f, 30.f});
    Tensor flat_indices(vec2<float>{{2.f, 0.f}, {1.f, 2.f}});
    check_tensor(flat.gather_flat(flat_indices), {2, 2}, {30.f, 10.f, 20.f, 30.f});
}

TEST_CASE("tensor serialization") {
    Tensor original(vec2<float>{{1.25f, 2.5f, 3.75f}, {4.25f, 5.5f, 6.75f}});
    Tensor non_contiguous = original.transpose();
    Tensor restored = Tensor::deserialize(non_contiguous.serialize());

    check_tensor(restored, {3, 2}, {1.25f, 4.25f, 2.5f, 5.5f, 3.75f, 6.75f});
    CHECK(restored.is_contiguous());
}

TEST_SUITE_END;

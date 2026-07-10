#include <doctest/doctest.h>
#include "emel/tensor.h"

// check a tensor's logical shape and values
static void check_tensor(
    const Tensor &actual,
    const vec<int> &expected_shape,
    const vec<double> &expected_data
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
    Tensor one_dimensional(vec<double>{1., 2., 3.});
    check_tensor(one_dimensional, {3}, {1., 2., 3.});

    Tensor two_dimensional(vec2<double>{{1., 2.}, {3., 4.}});
    check_tensor(two_dimensional, {2, 2}, {1., 2., 3., 4.});

    Tensor filled(vec<int>{2, 3}, 7.);
    check_tensor(filled, {2, 3}, {7., 7., 7., 7., 7., 7.});

    filled.at({1, 2}) = 9.;
    CHECK(filled.at({1, 2}) == 9.);
    CHECK(filled.is_contiguous());
}

TEST_CASE("tensor shape operations") {
    Tensor reshaped(vec<double>{1., 2., 3., 4., 5., 6.});
    reshaped.reshape({2, 3});
    check_tensor(reshaped, {2, 3}, {1., 2., 3., 4., 5., 6.});

    Tensor broadcasted(vec<double>{1., 2., 3.});
    broadcasted.broadcast({2, 3});
    check_tensor(broadcasted, {2, 3}, {1., 2., 3., 1., 2., 3.});
    CHECK_FALSE(broadcasted.is_contiguous());
    CHECK(broadcasted.materialize().is_contiguous());

    broadcasted.unbroadcast({3});
    check_tensor(broadcasted, {3}, {2., 4., 6.});

    Tensor axes(vec<double>{1., 2., 3., 4., 5., 6.});
    axes.reshape({1, 2, 3});
    check_tensor(axes.permute({2, 1, 0}), {3, 2, 1}, {1., 4., 2., 5., 3., 6.});
    check_tensor(axes.transpose(), {1, 3, 2}, {1., 4., 2., 5., 3., 6.});
}

TEST_CASE("tensor arithmetic and functionals") {
    Tensor matrix(vec2<double>{{1., 2., 3.}, {4., 5., 6.}});
    Tensor row(vec<double>{1., 2., 3.});

    check_tensor(matrix + row, {2, 3}, {2., 4., 6., 5., 7., 9.});
    check_tensor(matrix - row, {2, 3}, {0., 0., 0., 3., 3., 3.});
    check_tensor(matrix.hadamard(row), {2, 3}, {1., 4., 9., 4., 10., 18.});
    check_tensor(matrix.ediv(row), {2, 3}, {1., 1., 1., 4., 2.5, 2.});
    check_tensor(-row, {3}, {-1., -2., -3.});

    Tensor inplace = matrix;
    inplace += row;
    inplace -= row;
    check_tensor(inplace, {2, 3}, {1., 2., 3., 4., 5., 6.});

    check_tensor(row.apply([](double value) { return value * value; }), {3}, {1., 4., 9.});
    row.apply_inplace([](double value) { return value + 0.5; });
    check_tensor(row, {3}, {1.5, 2.5, 3.5});
}

TEST_CASE("tensor matrix multiplication") {
    Tensor left(vec2<double>{{1., 2., 3.}, {4., 5., 6.}});
    Tensor right(vec2<double>{{7., 8.}, {9., 10.}, {11., 12.}});

    check_tensor(left * right, {2, 2}, {58., 64., 139., 154.});
}

TEST_CASE("tensor reductions") {
    Tensor values(vec2<double>{{1., 5., 3.}, {4., 2., 6.}});

    check_tensor(values.sum(0, false), {3}, {5., 7., 9.});
    check_tensor(values.sum(1, true), {2, 1}, {9., 12.});
    check_tensor(values.max(0, true), {1, 3}, {4., 5., 6.});
    check_tensor(values.max(1, false), {2}, {5., 6.});
    check_tensor(values.argmax(1), {2, 3}, {0., 1., 0., 0., 0., 1.});
}

TEST_CASE("tensor gathering") {
    Tensor values(vec2<double>{{1., 2., 3.}, {4., 5., 6.}});
    Tensor indices(vec2<double>{{0., 1.}, {1., 2.}});
    check_tensor(values.gather(indices), {2}, {2., 6.});

    Tensor flat(vec<double>{10., 20., 30.});
    Tensor flat_indices(vec2<double>{{2., 0.}, {1., 2.}});
    check_tensor(flat.gather_flat(flat_indices), {2, 2}, {30., 10., 20., 30.});
}

TEST_CASE("tensor serialization") {
    Tensor original(vec2<double>{{1.25, 2.5, 3.75}, {4.25, 5.5, 6.75}});
    Tensor non_contiguous = original.transpose();
    Tensor restored = Tensor::deserialize(non_contiguous.serialize());

    check_tensor(restored, {3, 2}, {1.25, 4.25, 2.5, 5.5, 3.75, 6.75});
    CHECK(restored.is_contiguous());
}

TEST_SUITE_END;

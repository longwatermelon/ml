#include <doctest/doctest.h>
#include "emel/autograd.h"
#include <functional>

using Objective = std::function<GTensor(const GTensor&)>;

// construct a contiguous tensor from a shape and flat values
static Tensor shaped_tensor(const vec<int> &shape, const vec<double> &data) {
    Tensor out(shape, 0.);
    REQUIRE(out.num_el() == sz(data));
    out.data = data;
    return out;
}

// reduce every axis to produce a scalar
static GTensor sum_all(const GTensor &x) {
    GTensor out = x;
    for (int axis = 0; axis < sz(x.get_tensor().shape); ++axis) {
        out = out.sum_reduce(axis, true);
    }
    return out;
}

// compare autograd against central finite differences
static void check_gradient(const Tensor &initial, const Objective &objective) {
    GTensor x(initial);
    GTensor result = objective(x);
    REQUIRE(result.get_tensor().num_el() == 1);
    result.compute_all_grads();

    Tensor analytic = x.get_grad().materialize();
    Tensor base = initial.materialize();
    REQUIRE(analytic.shape == initial.shape);
    constexpr double eps = 1e-6;

    for (int i = 0; i < base.num_el(); ++i) {
        Tensor plus = base;
        Tensor minus = base;
        plus.data[i] += eps;
        minus.data[i] -= eps;

        double upper = objective(GTensor(plus)).get_tensor().materialize().data[0];
        double lower = objective(GTensor(minus)).get_tensor().materialize().data[0];
        double numerical = (upper - lower) / (2 * eps);

        CAPTURE(i);
        CAPTURE(analytic.data[i]);
        CAPTURE(numerical);
        CHECK(analytic.data[i] == doctest::Approx(numerical).epsilon(1e-5).scale(1.0));
    }
}

TEST_SUITE_BEGIN("autograd");

TEST_CASE("elementwise operation gradients match finite differences") {
    Tensor values(vec<double>{0.4, 1.2, 2.3});
    check_gradient(values, [](const GTensor &x) {
        return sum_all(x.exp());
    });
    check_gradient(values, [](const GTensor &x) {
        return sum_all(x.log());
    });
    check_gradient(values, [](const GTensor &x) {
        return sum_all(x.sqrt());
    });
    check_gradient(Tensor(vec<double>{-1.3, 0.4, 2.3}), [](const GTensor &x) {
        return sum_all(x.relu());
    });

    check_gradient(values, [](const GTensor &x) {
        GTensor constants(vec<double>{0.7, 1.5, 2.1});
        return sum_all(x.ediv(constants));
    });
    check_gradient(values, [](const GTensor &x) {
        GTensor constants(vec<double>{0.7, 1.5, 2.1});
        return sum_all(constants.ediv(x));
    });
}

TEST_CASE("shared graph gradients accumulate from every path") {
    Tensor values(vec<double>{-1.5, 0.2, 2.0});
    check_gradient(values, [](const GTensor &x) {
        return sum_all(x.hadamard(x) + x);
    });
}

TEST_CASE("broadcast gradients reduce to the original shape") {
    Tensor bias(vec<double>{0.2, -0.4, 1.1});
    check_gradient(bias, [](const GTensor &x) {
        GTensor matrix(vec2<double>{{1., 2., 3.}, {4., 5., 6.}});
        GTensor weights(vec2<double>{{0.5, -1., 2.}, {3., 0.25, -0.75}});
        return sum_all((matrix + x).hadamard(weights));
    });
}

TEST_CASE("broadcast gradients reduce across multiple singleton axes") {
    Tensor left = shaped_tensor(
        {2, 1, 3},
        {0.2, -0.4, 1.1, 0.7, 0.3, -0.8}
    );
    Tensor right = shaped_tensor(
        {1, 4, 1},
        {1.0, -0.5, 0.25, 2.0}
    );
    GTensor weights(shaped_tensor(
        {2, 4, 3},
        {1., -2., 0.5, 0.25, 3., -1., -0.75, 0.4, 2., 1.5, -0.2, 0.8,
         -1.1, 0.6, 0.3, 2.2, -0.9, 1.4, 0.7, -1.5, 0.2, 0.1, 1.8, -0.4}
    ));

    check_gradient(left, [&](const GTensor &x) {
        return sum_all((x + GTensor(right)).hadamard(weights));
    });
    check_gradient(right, [&](const GTensor &x) {
        return sum_all((GTensor(left) + x).hadamard(weights));
    });
}

TEST_CASE("matmul gradients match finite differences for both operands") {
    Tensor left(vec2<double>{{0.2, -0.3, 0.7}, {1.1, 0.4, -0.5}});
    Tensor right(vec2<double>{{0.6, -0.2}, {0.9, 0.3}, {-0.7, 1.2}});
    GTensor weights(vec2<double>{{1.5, -0.5}, {0.25, 2.0}});

    check_gradient(left, [&](const GTensor &x) {
        return sum_all((x * GTensor(right)).hadamard(weights));
    });
    check_gradient(right, [&](const GTensor &x) {
        return sum_all((GTensor(left) * x).hadamard(weights));
    });
}

TEST_CASE("batched matmul gradients handle broadcasted batch dimensions") {
    Tensor left = shaped_tensor(
        {2, 2, 3},
        {0.2, -0.3, 0.7, 1.1, 0.4, -0.5,
         -0.8, 0.6, 0.9, 0.3, -1.2, 0.5}
    );
    Tensor right = shaped_tensor(
        {1, 3, 2},
        {0.6, -0.2, 0.9, 0.3, -0.7, 1.2}
    );
    GTensor weights(shaped_tensor(
        {2, 2, 2},
        {1.5, -0.5, 0.25, 2.0, -1.0, 0.75, 0.4, -0.3}
    ));

    check_gradient(left, [&](const GTensor &x) {
        return sum_all((x * GTensor(right)).hadamard(weights));
    });
    check_gradient(right, [&](const GTensor &x) {
        return sum_all((GTensor(left) * x).hadamard(weights));
    });
}

TEST_CASE("reduction gradients match finite differences") {
    Tensor values(vec2<double>{{1., 4., -2.}, {3., -1., 2.}});

    check_gradient(values, [](const GTensor &x) {
        GTensor weights(vec<double>{2., -0.5});
        return sum_all(x.sum_reduce(1, false).hadamard(weights));
    });
    check_gradient(values, [](const GTensor &x) {
        GTensor weights(vec<double>{2., -0.5});
        return sum_all(x.max_reduce(1, false).hadamard(weights));
    });
}

TEST_CASE("reshape and permute gradients preserve element correspondence") {
    Tensor values(vec2<double>{{0.2, 0.4, 0.6}, {0.8, 1.0, 1.2}});
    check_gradient(values, [](const GTensor &x) {
        GTensor weights(vec2<double>{{1., -2., 0.5}, {3., 0.25, -1.}});
        GTensor transformed = x.reshape({3, 2}).permute({1, 0});
        return sum_all(transformed.hadamard(weights));
    });
}

TEST_CASE("gather gradients accumulate repeated indices") {
    Tensor values(vec<double>{0.3, -0.7, 1.4});
    check_gradient(values, [](const GTensor &x) {
        Tensor indices(vec2<double>{{1., 1.}, {2., 0.}});
        GTensor weights(vec2<double>{{0.5, 2.}, {-1., 3.}});
        return sum_all(x.gather_flat(indices).hadamard(weights));
    });
}

TEST_CASE("multi-axis gather gradients scatter coordinate tuples") {
    Tensor values(vec2<double>{{0.3, -0.7, 1.4}, {2.0, 0.5, -1.2}});
    check_gradient(values, [](const GTensor &x) {
        Tensor indices = shaped_tensor(
            {2, 2, 2},
            {0., 1., 0., 1., 1., 2., 1., 0.}
        );
        GTensor weights(vec2<double>{{0.5, 2.}, {-1., 3.}});
        return sum_all(x.gather(indices).hadamard(weights));
    });
}

TEST_CASE("softmax gradients match finite differences") {
    Tensor logits(vec2<double>{{0.2, 1.1, -0.4}, {2.0, -1.0, 0.5}});
    check_gradient(logits, [](const GTensor &x) {
        GTensor weights(vec2<double>{{1., -0.5, 2.}, {-1., 0.25, 0.75}});
        return sum_all(x.softmax(1).hadamard(weights));
    });
}

TEST_CASE("recomputing gradients clears previous values") {
    GTensor x(vec<double>{-1.5, 0.2, 2.0});
    GTensor result = sum_all(x.hadamard(x) + x);

    result.compute_all_grads();
    Tensor first = x.get_grad();
    result.compute_all_grads();
    Tensor second = x.get_grad();

    REQUIRE(first.shape == second.shape);
    for (int i = 0; i < first.num_el(); ++i) {
        CAPTURE(i);
        CHECK(second.data[i] == doctest::Approx(first.data[i]));
    }
}

TEST_SUITE_END;

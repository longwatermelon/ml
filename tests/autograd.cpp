#include <doctest/doctest.h>
#include "emel/autograd.h"
#include <functional>

using Objective = std::function<GTensor(const GTensor&)>;

// construct a contiguous tensor from a shape and flat values
static Tensor shaped_tensor(const vec<int> &shape, const vec<float> &data) {
    Tensor out(shape, 0.f);
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
    constexpr float eps = 1e-3f;

    for (int i = 0; i < base.num_el(); ++i) {
        Tensor plus = base;
        Tensor minus = base;
        plus.data[i] += eps;
        minus.data[i] -= eps;

        float upper = objective(GTensor(plus)).get_tensor().materialize().data[0];
        float lower = objective(GTensor(minus)).get_tensor().materialize().data[0];
        float numerical = (upper - lower) / (2 * eps);

        CAPTURE(i);
        CAPTURE(analytic.data[i]);
        CAPTURE(numerical);
        CHECK(analytic.data[i] == doctest::Approx(numerical).epsilon(1e-3f).scale(1.0f));
    }
}

TEST_SUITE_BEGIN("autograd");

TEST_CASE("elementwise operation gradients match finite differences") {
    Tensor values(vec<float>{0.4f, 1.2f, 2.3f});
    check_gradient(values, [](const GTensor &x) {
        return sum_all(x.exp());
    });
    check_gradient(values, [](const GTensor &x) {
        return sum_all(x.log());
    });
    check_gradient(values, [](const GTensor &x) {
        return sum_all(x.sqrt());
    });
    check_gradient(Tensor(vec<float>{-1.3f, 0.4f, 2.3f}), [](const GTensor &x) {
        return sum_all(x.relu());
    });

    check_gradient(values, [](const GTensor &x) {
        GTensor constants(vec<float>{0.7f, 1.5f, 2.1f});
        return sum_all(x.ediv(constants));
    });
    check_gradient(values, [](const GTensor &x) {
        GTensor constants(vec<float>{0.7f, 1.5f, 2.1f});
        return sum_all(constants.ediv(x));
    });
}

TEST_CASE("shared graph gradients accumulate from every path") {
    Tensor values(vec<float>{-1.5f, 0.2f, 2.0f});
    check_gradient(values, [](const GTensor &x) {
        return sum_all(x.hadamard(x) + x);
    });
}

TEST_CASE("broadcast gradients reduce to the original shape") {
    Tensor bias(vec<float>{0.2f, -0.4f, 1.1f});
    check_gradient(bias, [](const GTensor &x) {
        GTensor matrix(vec2<float>{{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}});
        GTensor weights(vec2<float>{{0.5f, -1.f, 2.f}, {3.f, 0.25f, -0.75f}});
        return sum_all((matrix + x).hadamard(weights));
    });
}

TEST_CASE("broadcast gradients reduce across multiple singleton axes") {
    Tensor left = shaped_tensor(
        {2, 1, 3},
        {0.2f, -0.4f, 1.1f, 0.7f, 0.3f, -0.8f}
    );
    Tensor right = shaped_tensor(
        {1, 4, 1},
        {1.0f, -0.5f, 0.25f, 2.0f}
    );
    GTensor weights(shaped_tensor(
        {2, 4, 3},
        {1.f, -2.f, 0.5f, 0.25f, 3.f, -1.f, -0.75f, 0.4f, 2.f, 1.5f, -0.2f, 0.8f,
         -1.1f, 0.6f, 0.3f, 2.2f, -0.9f, 1.4f, 0.7f, -1.5f, 0.2f, 0.1f, 1.8f, -0.4f}
    ));

    check_gradient(left, [&](const GTensor &x) {
        return sum_all((x + GTensor(right)).hadamard(weights));
    });
    check_gradient(right, [&](const GTensor &x) {
        return sum_all((GTensor(left) + x).hadamard(weights));
    });
}

TEST_CASE("matmul gradients match finite differences for both operands") {
    Tensor left(vec2<float>{{0.2f, -0.3f, 0.7f}, {1.1f, 0.4f, -0.5f}});
    Tensor right(vec2<float>{{0.6f, -0.2f}, {0.9f, 0.3f}, {-0.7f, 1.2f}});
    GTensor weights(vec2<float>{{1.5f, -0.5f}, {0.25f, 2.0f}});

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
        {0.2f, -0.3f, 0.7f, 1.1f, 0.4f, -0.5f,
         -0.8f, 0.6f, 0.9f, 0.3f, -1.2f, 0.5f}
    );
    Tensor right = shaped_tensor(
        {1, 3, 2},
        {0.6f, -0.2f, 0.9f, 0.3f, -0.7f, 1.2f}
    );
    GTensor weights(shaped_tensor(
        {2, 2, 2},
        {1.5f, -0.5f, 0.25f, 2.0f, -1.0f, 0.75f, 0.4f, -0.3f}
    ));

    check_gradient(left, [&](const GTensor &x) {
        return sum_all((x * GTensor(right)).hadamard(weights));
    });
    check_gradient(right, [&](const GTensor &x) {
        return sum_all((GTensor(left) * x).hadamard(weights));
    });
}

TEST_CASE("reduction gradients match finite differences") {
    Tensor values(vec2<float>{{1.f, 4.f, -2.f}, {3.f, -1.f, 2.f}});

    check_gradient(values, [](const GTensor &x) {
        GTensor weights(vec<float>{2.f, -0.5f});
        return sum_all(x.sum_reduce(1, false).hadamard(weights));
    });
    check_gradient(values, [](const GTensor &x) {
        GTensor weights(vec<float>{2.f, -0.5f});
        return sum_all(x.max_reduce(1, false).hadamard(weights));
    });
}

TEST_CASE("reshape and permute gradients preserve element correspondence") {
    Tensor values(vec2<float>{{0.2f, 0.4f, 0.6f}, {0.8f, 1.0f, 1.2f}});
    check_gradient(values, [](const GTensor &x) {
        GTensor weights(vec2<float>{{1.f, -2.f, 0.5f}, {3.f, 0.25f, -1.f}});
        GTensor transformed = x.reshape({3, 2}).permute({1, 0});
        return sum_all(transformed.hadamard(weights));
    });
}

TEST_CASE("gather gradients accumulate repeated indices") {
    Tensor values(vec<float>{0.3f, -0.7f, 1.4f});
    check_gradient(values, [](const GTensor &x) {
        Tensor indices(vec2<float>{{1.f, 1.f}, {2.f, 0.f}});
        GTensor weights(vec2<float>{{0.5f, 2.f}, {-1.f, 3.f}});
        return sum_all(x.gather_flat(indices).hadamard(weights));
    });
}

TEST_CASE("multi-axis gather gradients scatter coordinate tuples") {
    Tensor values(vec2<float>{{0.3f, -0.7f, 1.4f}, {2.0f, 0.5f, -1.2f}});
    check_gradient(values, [](const GTensor &x) {
        Tensor indices = shaped_tensor(
            {2, 2, 2},
            {0.f, 1.f, 0.f, 1.f, 1.f, 2.f, 1.f, 0.f}
        );
        GTensor weights(vec2<float>{{0.5f, 2.f}, {-1.f, 3.f}});
        return sum_all(x.gather(indices).hadamard(weights));
    });
}

TEST_CASE("softmax gradients match finite differences") {
    Tensor logits(vec2<float>{{0.2f, 1.1f, -0.4f}, {2.0f, -1.0f, 0.5f}});
    check_gradient(logits, [](const GTensor &x) {
        GTensor weights(vec2<float>{{1.f, -0.5f, 2.f}, {-1.f, 0.25f, 0.75f}});
        return sum_all(x.softmax(1).hadamard(weights));
    });
}

TEST_CASE("recomputing gradients clears previous values") {
    GTensor x(vec<float>{-1.5f, 0.2f, 2.0f});
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

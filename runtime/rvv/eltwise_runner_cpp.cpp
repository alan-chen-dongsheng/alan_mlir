#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <array>

// MLIR-generated functions (C++ linkage)
void eltwise_add(float *lhs, float *rhs, float *out);
void eltwise_mul(float *lhs, float *rhs, float *out);
void eltwise_max(float *lhs, float *rhs, float *out);

constexpr int N = 16;

static int test_add() {
  std::array<float, N> lhs, rhs, out;
  for (int i = 0; i < N; i++) {
    lhs[i] = static_cast<float>(i);
    rhs[i] = static_cast<float>(i * 10);
  }
  eltwise_add(lhs.data(), rhs.data(), out.data());

  int failed = 0;
  for (int i = 0; i < N; i++) {
    float expected = lhs[i] + rhs[i];
    if (std::fabs(out[i] - expected) > 1e-5f) {
      std::printf("ADD FAILED at %d: got %f, expected %f\n", i, out[i], expected);
      failed = 1;
    }
  }
  if (!failed) std::printf("ADD PASSED\n");
  return failed;
}

static int test_mul() {
  std::array<float, N> lhs, rhs, out;
  for (int i = 0; i < N; i++) {
    lhs[i] = static_cast<float>(i + 1);
    rhs[i] = static_cast<float>(i + 2);
  }
  eltwise_mul(lhs.data(), rhs.data(), out.data());

  int failed = 0;
  for (int i = 0; i < N; i++) {
    float expected = lhs[i] * rhs[i];
    if (std::fabs(out[i] - expected) > 1e-5f) {
      std::printf("MUL FAILED at %d: got %f, expected %f\n", i, out[i], expected);
      failed = 1;
    }
  }
  if (!failed) std::printf("MUL PASSED\n");
  return failed;
}

static int test_max() {
  std::array<float, N> lhs, rhs, out;
  for (int i = 0; i < N; i++) {
    lhs[i] = static_cast<float>(i);
    rhs[i] = static_cast<float>(N - i);
  }
  eltwise_max(lhs.data(), rhs.data(), out.data());

  int failed = 0;
  for (int i = 0; i < N; i++) {
    float expected = (lhs[i] > rhs[i]) ? lhs[i] : rhs[i];
    if (std::fabs(out[i] - expected) > 1e-5f) {
      std::printf("MAX FAILED at %d: got %f, expected %f\n", i, out[i], expected);
      failed = 1;
    }
  }
  if (!failed) std::printf("MAX PASSED\n");
  return failed;
}

int main() {
  std::printf("Running Alan Eltwise RVV (C++ source) Tests on Spike...\n");
  int failures = 0;
  failures += test_add();
  failures += test_mul();
  failures += test_max();

  if (failures == 0) {
    std::printf("\nAll RVV tests PASSED!\n");
    return 0;
  } else {
    std::printf("\n%d RVV test(s) FAILED!\n", failures);
    return 1;
  }
}

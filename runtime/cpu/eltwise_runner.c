#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// MemRef descriptor returned by MLIR functions
typedef struct {
  void *data;
  void *aligned_data;
  long offset;
  long sizes[1];
  long strides[1];
} MemRef1Df32Result;

// Forward declaration of the compiled MLIR functions
MemRef1Df32Result eltwise_add(
  void *lhs_data, void *lhs_aligned, long lhs_offset, long lhs_size, long lhs_stride,
  void *rhs_data, void *rhs_aligned, long rhs_offset, long rhs_size, long rhs_stride);

MemRef1Df32Result eltwise_mul(
  void *lhs_data, void *lhs_aligned, long lhs_offset, long lhs_size, long lhs_stride,
  void *rhs_data, void *rhs_aligned, long rhs_offset, long rhs_size, long rhs_stride);

MemRef1Df32Result eltwise_max(
  void *lhs_data, void *lhs_aligned, long lhs_offset, long lhs_size, long lhs_stride,
  void *rhs_data, void *rhs_aligned, long rhs_offset, long rhs_size, long rhs_stride);

// Test add operation
int test_add() {
  const long n = 4;
  float *lhs = (float *)malloc(n * sizeof(float));
  float *rhs = (float *)malloc(n * sizeof(float));

  // Initialize
  for (long i = 0; i < n; i++) {
    lhs[i] = (float)i;
    rhs[i] = (float)(i * 10);
  }

  // Call MLIR function
  MemRef1Df32Result result = eltwise_add(
    lhs, lhs, 0L, n, 1L,
    rhs, rhs, 0L, n, 1L);

  // Verify
  int failed = 0;
  float *result_data = (float *)result.aligned_data;
  for (long i = 0; i < n; i++) {
    float expected = lhs[i] + rhs[i];
    if (fabsf(result_data[i] - expected) > 1e-5f) {
      printf("ADD FAILED at index %ld: got %f, expected %f\n",
             i, result_data[i], expected);
      failed = 1;
    }
  }

  if (!failed) {
    printf("ADD PASSED\n");
  }

  free(lhs);
  free(rhs);
  free(result.data);
  return failed;
}

// Test mul operation
int test_mul() {
  const long n = 4;
  float *lhs = (float *)malloc(n * sizeof(float));
  float *rhs = (float *)malloc(n * sizeof(float));

  // Initialize
  for (long i = 0; i < n; i++) {
    lhs[i] = (float)(i + 1);
    rhs[i] = (float)(i + 2);
  }

  // Call MLIR function
  MemRef1Df32Result result = eltwise_mul(
    lhs, lhs, 0L, n, 1L,
    rhs, rhs, 0L, n, 1L);

  // Verify
  int failed = 0;
  float *result_data = (float *)result.aligned_data;
  for (long i = 0; i < n; i++) {
    float expected = lhs[i] * rhs[i];
    if (fabsf(result_data[i] - expected) > 1e-5f) {
      printf("MUL FAILED at index %ld: got %f, expected %f\n",
             i, result_data[i], expected);
      failed = 1;
    }
  }

  if (!failed) {
    printf("MUL PASSED\n");
  }

  free(lhs);
  free(rhs);
  free(result.data);
  return failed;
}

// Test max operation
int test_max() {
  const long n = 4;
  float *lhs = (float *)malloc(n * sizeof(float));
  float *rhs = (float *)malloc(n * sizeof(float));

  // Initialize with crossing values
  lhs[0] = 1.0f; lhs[1] = 5.0f; lhs[2] = 3.0f; lhs[3] = 10.0f;
  rhs[0] = 3.0f; rhs[1] = 2.0f; rhs[2] = 7.0f; rhs[3] = 4.0f;

  // Call MLIR function
  MemRef1Df32Result result = eltwise_max(
    lhs, lhs, 0L, n, 1L,
    rhs, rhs, 0L, n, 1L);

  // Verify
  float expected[] = {3.0f, 5.0f, 7.0f, 10.0f};
  int failed = 0;
  float *result_data = (float *)result.aligned_data;
  for (long i = 0; i < n; i++) {
    if (fabsf(result_data[i] - expected[i]) > 1e-5f) {
      printf("MAX FAILED at index %ld: got %f, expected %f\n",
             i, result_data[i], expected[i]);
      failed = 1;
    }
  }

  if (!failed) {
    printf("MAX PASSED\n");
  }

  free(lhs);
  free(rhs);
  free(result.data);
  return failed;
}

int main(int argc, char **argv) {
  int failures = 0;

  printf("Running Alan Eltwise CPU Tests...\n");

  failures += test_add();
  failures += test_mul();
  failures += test_max();

  if (failures == 0) {
    printf("\nAll tests PASSED!\n");
    return 0;
  } else {
    printf("\n%d test(s) FAILED!\n", failures);
    return 1;
  }
}

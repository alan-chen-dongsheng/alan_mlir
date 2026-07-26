// RUN: alan-opt %s | FileCheck %s

// CHECK-LABEL: func.func @relu_f32
func.func @relu_f32(%input: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: alan.relu
  // CHECK-SAME: tensor<4xf32>
  %result = alan.relu %input : tensor<4xf32> -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// CHECK-LABEL: func.func @relu_rank2
func.func @relu_rank2(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: alan.relu
  // CHECK-SAME: tensor<2x3xf32>
  %result = alan.relu %input : tensor<2x3xf32> -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

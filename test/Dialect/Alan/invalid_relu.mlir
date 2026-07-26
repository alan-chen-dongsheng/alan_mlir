// RUN: alan-opt %s 2>&1 | FileCheck %s

// CHECK: error: 'alan.relu' op requires the same type for all operands and results
func.func @shape_mismatch(%input: tensor<4xf32>) -> tensor<8xf32> {
  %result = alan.relu %input : tensor<4xf32> -> tensor<8xf32>
  return %result : tensor<8xf32>
}

// CHECK: error: 'alan.relu' op requires the same type for all operands and results
func.func @type_mismatch(%input: tensor<4xf32>) -> tensor<4xi32> {
  %result = alan.relu %input : tensor<4xf32> -> tensor<4xi32>
  return %result : tensor<4xi32>
}

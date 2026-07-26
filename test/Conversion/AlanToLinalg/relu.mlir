// RUN: alan-opt %s --convert-alan-to-linalg | FileCheck %s

// CHECK-LABEL: func.func @relu_f32
func.func @relu_f32(%input: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: tensor.empty
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#map, #map]
  // CHECK-SAME: iterator_types = ["parallel"]
  // CHECK: arith.constant 0.000000e+00
  // CHECK: arith.maximumf
  // CHECK: linalg.yield
  %result = alan.relu %input : tensor<4xf32> -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// CHECK-LABEL: func.func @relu_rank2
func.func @relu_rank2(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: indexing_maps = [#map1, #map1]
  // CHECK-SAME: iterator_types = ["parallel", "parallel"]
  %result = alan.relu %input : tensor<2x3xf32> -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}

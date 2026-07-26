func.func @eltwise_add(%lhs: tensor<16xf32>, %rhs: tensor<16xf32>) -> tensor<16xf32> {
  %result = alan.eltwise %lhs, %rhs {kind = "add"} : tensor<16xf32>, tensor<16xf32> -> tensor<16xf32>
  return %result : tensor<16xf32>
}

func.func @eltwise_mul(%lhs: tensor<16xf32>, %rhs: tensor<16xf32>) -> tensor<16xf32> {
  %result = alan.eltwise %lhs, %rhs {kind = "mul"} : tensor<16xf32>, tensor<16xf32> -> tensor<16xf32>
  return %result : tensor<16xf32>
}

func.func @eltwise_max(%lhs: tensor<16xf32>, %rhs: tensor<16xf32>) -> tensor<16xf32> {
  %result = alan.eltwise %lhs, %rhs {kind = "max"} : tensor<16xf32>, tensor<16xf32> -> tensor<16xf32>
  return %result : tensor<16xf32>
}

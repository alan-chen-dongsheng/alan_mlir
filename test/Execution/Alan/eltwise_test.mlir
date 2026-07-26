func.func @eltwise_add(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
  %result = alan.eltwise %lhs, %rhs {kind = "add"} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  return %result : tensor<4xf32>
}

func.func @eltwise_mul(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
  %result = alan.eltwise %lhs, %rhs {kind = "mul"} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  return %result : tensor<4xf32>
}

func.func @eltwise_max(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
  %result = alan.eltwise %lhs, %rhs {kind = "max"} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  return %result : tensor<4xf32>
}

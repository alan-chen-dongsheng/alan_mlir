# Alan MLIR Lowering Pipelines

本文档详细介绍 Alan Dialect 到 CPU 和 RVV 后端的完整 lowering 流程。

## 目录结构

- [CPU Lowering Pipeline](./cpu_pipeline.md) - 从 Alan 到本地 CPU 可执行文件
- [RVV Lowering Pipeline](./rvv_pipeline.md) - 从 Alan 到 RISC-V RVV ELF
- [Pipeline Comparison](./pipeline_comparison.md) - CPU 与 RVV 路径对比

## 概览

Alan MLIR 实现了双后端架构,共享前半部分的 lowering 路径:

```
Alan Dialect (alan.eltwise)
    ↓
[共享路径]
    ↓
Linalg Dialect (linalg.generic)
    ↓
Bufferization (MemRef)
    ↓
SCF/Control Flow
    ↓
[分支点]
    ↓           ↓
CPU 路径      RVV 路径
    ↓           ↓
LLVM Dialect  LLVM Dialect
    ↓           ↓
LLVM IR       LLVM IR
    ↓           ↓
Native ELF    RV64GCV ELF
    ↓           ↓
本地执行      Spike 模拟
```

## 关键概念

### 1. Dialect Conversion

MLIR 使用 dialect conversion 框架进行 IR 转换:
- **Source Dialect**: 要转换的方言 (如 Alan)
- **Target Dialect**: 目标方言 (如 Linalg, LLVM)
- **Conversion Pattern**: 定义如何将 source op 转换为目标 op

### 2. Bufferization

将 tensor (值语义) 转换为 memref (引用语义):
- **Tensor**: 不可变,函数式语义
- **MemRef**: 可变,内存缓冲区
- **One-Shot Bufferize**: MLIR 的 bufferization pass

### 3. Lowering Strategy

采用渐进式 lowering:
1. **High-level**: 抽象的张量操作 (Alan, Linalg)
2. **Mid-level**: 循环和内存访问 (SCF, MemRef)
3. **Low-level**: 控制流和基本操作 (CF, Arith)
4. **Target**: LLVM Dialect → LLVM IR → 机器码

## 测试文件

所有示例基于以下测试文件:

```mlir
func.func @eltwise_add(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
  %result = alan.eltwise %lhs, %rhs {kind = "add"} 
    : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  return %result : tensor<4xf32>
}
```

## 运行示例

### CPU Pipeline

```bash
./tools/run_alan_cpu.sh test/Execution/Alan/eltwise_test.mlir
```

### RVV Pipeline

```bash
./tools/run_alan_rvv_spike.sh test/Execution/Alan/eltwise_rvv_test.mlir
```

## 调试技巧

保留中间产物:

```bash
KEEP_TEMP=1 ./tools/run_alan_cpu.sh test.mlir
ls /tmp/alan_cpu_*/
```

查看每个步骤的 IR:
- `step1_linalg.mlir` - Linalg dialect
- `step2_bufferized.mlir` - Bufferized (MemRef)
- `step3_loops.mlir` - SCF loops
- `step4_llvm.mlir` - LLVM dialect
- `module.ll` - LLVM IR
- `module.o` - 目标文件

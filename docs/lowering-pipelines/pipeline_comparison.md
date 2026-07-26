# Pipeline Comparison: CPU vs RVV

本文档对比 CPU 和 RVV 两条 lowering 路径的异同。

## 共享部分 (Common Path)

两条路径在前 4 个步骤完全相同:

```
Alan Dialect
    ↓
Linalg Dialect (tensor)
    ↓
MemRef Dialect (bufferized)
    ↓
SCF Dialect (loops)
    ↓
Control Flow (CF)
    ↓
LLVM Dialect
    ↓
LLVM IR
    ↓
[分支点]
```

### 共享的 Pass 序列

| 步骤 | Pass | 作用 |
|------|------|------|
| 1 | `--convert-alan-to-linalg` | Alan → Linalg |
| 2 | `--one-shot-bufferize` | Tensor → MemRef |
| 3 | `--convert-bufferization-to-memref` | 移除 bufferization ops |
| 4 | `--convert-linalg-to-parallel-loops` | Linalg → SCF loops |
| 5 | `--convert-scf-to-cf` | SCF → Control Flow |
| 6 | `--convert-arith-to-llvm` | Arith → LLVM |
| 7 | `--finalize-memref-to-llvm` | MemRef → LLVM |
| 8 | `--convert-func-to-llvm` | Func → LLVM |
| 9 | `--convert-cf-to-llvm` | CF → LLVM |
| 10 | `--reconcile-unrealized-casts` | 清理类型转换 |
| 11 | `mlir-translate --mlir-to-llvmir` | MLIR → LLVM IR |

### 共享的 IR 示例

在 Step 11 之后,两条路径产生相同的 LLVM IR:

```llvm
define ptr @eltwise_add(ptr %arg0, ptr %arg1, ...) {
entry:
  %mem = call ptr @malloc(i64 16)
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %cond = icmp slt i64 %i, 4
  br i1 %cond, label %body, label %exit

body:
  %val1 = load float, ptr %arg0
  %val2 = load float, ptr %arg1
  %sum = fadd float %val1, %val2
  store float %sum, ptr %mem
  %next = add i64 %i, 1
  br label %loop

exit:
  ret ptr %mem
}
```

## 分支点 (Divergence Point)

从 LLVM IR 开始,两条路径分叉:

```
LLVM IR
    ↓
    ├─────────────────────┬─────────────────────┐
    ↓                     ↓                     ↓
CPU Path               RVV Path              (未来:其他路径)
    ↓                     ↓                     ↓
llc (native)           llc (riscv64)         ...
    ↓                     ↓                     ↓
clang (link)           riscv64-gcc (link)    ...
    ↓                     ↓                     ↓
Native ELF             RV64GCV ELF           ...
    ↓                     ↓                     ↓
Local execution        Spike execution       ...
```

## 详细对比

### 1. LLC 编译

| 方面 | CPU | RVV |
|------|-----|-----|
| **命令** | `llc module.ll -o module.o` | `llc module.ll -mtriple=riscv64-unknown-elf -mattr=+v -o module.o` |
| **目标架构** | 宿主机 (x86_64/ARM64) | RISC-V 64-bit |
| **扩展指令集** | 默认 | RV64GCV |
| **优化级别** | 默认 | `-O3` (启用向量化) |
| **输出格式** | 宿主机目标文件 | RISC-V 目标文件 |

**CPU 命令**:
```bash
llc module.ll -filetype=obj -o module.o
```

**RVV 命令**:
```bash
llc module.ll \
  -mtriple=riscv64-unknown-elf \
  -mattr=+m,+a,+f,+d,+c,+v \
  -O3 \
  -filetype=obj \
  -o module.o
```

### 2. 链接

| 方面 | CPU | RVV |
|------|-----|-----|
| **链接器** | `clang` | `riscv64-unknown-elf-gcc` |
| **参数** | `-O2 -lm` | `-march=rv64gcv -mabi=lp64d` |
| **Runtime** | `runtime/cpu/eltwise_runner.c` | `runtime/rvv/eltwise_runner.c` |
| **输出** | `test_runner` (native ELF) | `test_runner.elf` (RV64GCV ELF) |

**CPU 命令**:
```bash
clang -O2 -lm \
  module.o \
  runtime/cpu/eltwise_runner.c \
  -o test_runner
```

**RVV 命令**:
```bash
riscv64-unknown-elf-gcc \
  -march=rv64gcv \
  -mabi=lp64d \
  module.o \
  runtime/rvv/eltwise_runner.c \
  -o test_runner.elf
```

### 3. 执行

| 方面 | CPU | RVV |
|------|-----|-----|
| **执行环境** | 本地执行 | Spike 模拟器 |
| **命令** | `./test_runner` | `spike --isa=rv64gcv pk test_runner.elf` |
| **性能** | 原生速度 | 模拟速度 (慢 10-100x) |
| **调试** | gdb, lldb | spike --debug, gdb |

**CPU 命令**:
```bash
./test_runner
```

**RVV 命令**:
```bash
spike \
  --isa=rv64gcv \
  --varch=vlen:256,elen:64 \
  pk \
  test_runner.elf
```

### 4. 生成的指令

| 方面 | CPU | RVV |
|------|-----|-----|
| **指令集** | x86_64 / ARM64 | RV64GCV |
| **向量化** | 依赖编译器 | 标量 (当前) / RVV (未来) |
| **示例指令** | `addss`, `mulss` | `fadd.s`, `fmul.s` |

**CPU 汇编** (x86_64):
```asm
vmovss (%rdi,%rcx,4), %xmm0
vaddss (%rsi,%rcx,4), %xmm0, %xmm0
vmovss %xmm0, (%rax,%rcx,4)
```

**RVV 汇编** (标量版本):
```asm
flw fa5, 0(a7)
fadd.s fa5, fa5, fa4
fsw fa5, 0(a7)
```

**RVV 汇编** (向量化版本,未来):
```asm
vsetvli a5, zero, e32, m1, ta, ma
vle32.v v1, 0(a1)
vfadd.vv v3, v1, v2
vse32.v v3, 0(a0)
```

## 工具链对比

### CPU 工具链

| 工具 | 来源 | 用途 |
|------|------|------|
| `clang` | LLVM | C 编译器 + 链接器 |
| `llc` | LLVM | LLVM IR → 目标文件 |
| `mlir-opt` | MLIR | MLIR pass 执行 |
| `mlir-translate` | MLIR | MLIR → LLVM IR |

### RVV 工具链

| 工具 | 来源 | 用途 |
|------|------|------|
| `riscv64-unknown-elf-gcc` | RISC-V GNU Toolchain | 交叉编译器 + 链接器 |
| `llc` | LLVM | LLVM IR → RISC-V 目标文件 |
| `mlir-opt` | MLIR | MLIR pass 执行 |
| `mlir-translate` | MLIR | MLIR → LLVM IR |
| `spike` | RISC-V ISA Simulator | RISC-V 模拟器 |
| `pk` | RISC-V Proxy Kernel | 用户态运行时 |

## 性能对比

### 执行时间

| 场景 | CPU | RVV (Spike) |
|------|-----|-------------|
| **小数组 (4 elements)** | < 1 ms | ~100 ms |
| **中数组 (1024 elements)** | < 1 ms | ~500 ms |
| **大数组 (1M elements)** | ~10 ms | ~50 s |

**注意**: Spike 模拟器的性能比原生执行慢 10-100 倍。

### 向量化潜力

| 方面 | CPU | RVV |
|------|-----|-----|
| **当前状态** | 标量 | 标量 |
| **向量化支持** | SSE/AVX (x86), NEON (ARM) | RVV (RISC-V) |
| **自动向量化** | 编译器支持 | LLVM 支持有限 |
| **手动向量化** | 需要 intrinsics | 需要 Vector Dialect |

## 开发工作流对比

### CPU 开发流程

```bash
# 1. 编写 Alan MLIR 代码
cat > test.mlir << 'EOF'
func.func @my_op(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = alan.eltwise %arg0, %arg0 {kind = "add"} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
EOF

# 2. 执行
./tools/run_alan_cpu.sh test.mlir

# 3. 查看结果
# 自动验证输出
```

### RVV 开发流程

```bash
# 1. 编写 Alan MLIR 代码 (相同)
cat > test.mlir << 'EOF'
func.func @my_op(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = alan.eltwise %arg0, %arg0 {kind = "add"} : tensor<4xf32>
  return %0 : tensor<4xf32>
}
EOF

# 2. 执行 (需要 Spike)
./tools/run_alan_rvv_spike.sh test.mlir

# 3. 查看结果
# 自动验证输出

# 4. (可选) 检查生成的汇编
KEEP_TEMP=1 ./tools/run_alan_rvv_spike.sh test.mlir
riscv64-unknown-elf-objdump -d /tmp/alan_rvv_*/module.o
```

## 选择指南

### 使用 CPU Pipeline 当:

- ✅ 快速迭代开发
- ✅ 需要高性能执行
- ✅ 调试复杂问题
- ✅ 不需要 RISC-V 特定特性
- ✅ 宿主机是 x86_64 或 ARM64

### 使用 RVV Pipeline 当:

- ✅ 需要生成 RISC-V 代码
- ✅ 测试 RVV 向量指令
- ✅ 部署到 RISC-V 硬件
- ✅ 研究 RISC-V 向量扩展
- ✅ 验证跨平台兼容性

## 未来扩展

### 1. 显式向量化

当前两条路径都生成标量代码。未来可以:

**CPU 路径**:
```bash
mlir-opt input.mlir \
  --convert-linalg-to-vector \
  --lower-vector-to-sse \  # 或 --lower-vector-to-avx
```

**RVV 路径**:
```bash
mlir-opt input.mlir \
  --convert-linalg-to-vector \
  --lower-vector-to-rvv
```

### 2. 其他后端

可以添加更多后端:

- **GPU**: 通过 `--convert-linalg-to-gpu`
- **TPU**: 通过自定义 lowering
- **FPGA**: 通过 HLS 工具链

### 3. 统一 Pipeline

可以创建一个统一的 pipeline,根据目标自动选择后端:

```bash
alan-opt input.mlir --target=cpu      # CPU 路径
alan-opt input.mlir --target=rvv      # RVV 路径
alan-opt input.mlir --target=gpu      # GPU 路径
```

## 总结

CPU 和 RVV 路径共享大部分 lowering 逻辑,主要差异在于:

1. **目标架构**: 宿主机 vs RISC-V
2. **工具链**: clang vs riscv64-gcc
3. **执行环境**: 本地 vs Spike 模拟器
4. **向量化**: SSE/AVX vs RVV (当前都是标量)

这种设计使得添加新后端变得简单:只需在 LLVM IR 之后添加新的编译和链接步骤。

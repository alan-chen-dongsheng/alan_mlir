# RVV Lowering Pipeline

本文档详细介绍从 Alan Dialect 到 RISC-V RVV (Vector Extension) ELF 的完整 lowering 流程。

## Pipeline 概览

```
Alan Dialect
    ↓ --convert-alan-to-linalg
Linalg Dialect (tensor 语义)
    ↓ --one-shot-bufferize
Linalg Dialect (memref 语义)
    ↓ --convert-bufferization-to-memref
MemRef Operations
    ↓ --convert-linalg-to-parallel-loops
SCF Parallel Loops
    ↓ --convert-scf-to-cf
Control Flow (CF)
    ↓ --convert-arith-to-llvm, --convert-func-to-llvm, etc.
LLVM Dialect
    ↓ mlir-translate --mlir-to-llvmir
LLVM IR
    ↓ llc -mtriple=riscv64-unknown-elf -mattr=+v
RV64GCV Object File
    ↓ riscv64-unknown-elf-gcc (link)
RV64GCV ELF
    ↓ spike + pk
RISC-V Simulator Execution
```

## 与 CPU Pipeline 的区别

| 方面 | CPU Pipeline | RVV Pipeline |
|------|--------------|--------------|
| **目标架构** | 宿主机 (x86_64/ARM64) | RISC-V RV64GCV |
| **向量化** | 依赖编译器自动向量化 | 使用 RVV 向量指令 |
| **链接器** | clang | riscv64-unknown-elf-gcc |
| **执行环境** | 本地执行 | Spike 模拟器 |
| **LLC 参数** | 默认 | `-mtriple=riscv64-unknown-elf -mattr=+v` |

## 详细步骤

### Step 1-4: 与 CPU Pipeline 相同

前四个步骤与 CPU pipeline 完全相同:

1. **Alan → Linalg**: `--convert-alan-to-linalg`
2. **Bufferization**: `--one-shot-bufferize`
3. **Linalg → Loops**: `--convert-linalg-to-parallel-loops`
4. **→ LLVM Dialect**: `--convert-*-to-llvm`

**关键**: 在这个阶段,CPU 和 RVV 路径共享相同的 IR。差异从 Step 5 开始。

---

### Step 5: LLVM Dialect → LLVM IR

**命令**:
```bash
mlir-translate step4_llvm.mlir --mlir-to-llvmir -o module.ll
```

**输出**: LLVM IR (与 CPU 路径相同)

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
  %ptr1 = getelementptr float, ptr %arg0, i64 %i
  %val1 = load float, ptr %ptr1
  %ptr2 = getelementptr float, ptr %arg1, i64 %i
  %val2 = load float, ptr %ptr2
  %sum = fadd float %val1, %val2
  %ptr_out = getelementptr float, ptr %mem, i64 %i
  store float %sum, ptr %ptr_out
  %next = add i64 %i, 1
  br label %loop

exit:
  ret ptr %mem
}
```

---

### Step 6: LLVM IR → RV64GCV Object File

**命令**:
```bash
llc module.ll \
  -mtriple=riscv64-unknown-elf \
  -mattr=+m,+a,+f,+d,+c,+v \
  -O3 \
  -filetype=obj \
  -o module.o
```

**关键参数**:
- `-mtriple=riscv64-unknown-elf`: 目标架构为 RISC-V 64-bit
- `-mattr=+m,+a,+f,+d,+c,+v`: 启用 RISC-V 扩展
  - `+m`: 整数乘除法
  - `+a`: 原子操作
  - `+f`: 单精度浮点
  - `+d`: 双精度浮点
  - `+c`: 压缩指令
  - `+v`: **向量扩展 (RVV)**
- `-O3`: 最高优化级别 (启用自动向量化)

**输出**: RISC-V 目标文件

**查看生成的汇编**:
```bash
riscv64-unknown-elf-objdump -d module.o > disasm.txt
cat disasm.txt
```

**示例输出** (标量版本):
```asm
<eltwise_add>:
  addi sp, sp, -32
  sd ra, 24(sp)
  sd s0, 16(sp)
  addi s0, sp, 32
  
  ; 调用 malloc
  li a0, 16
  call malloc
  
  ; 循环
  li a5, 0
.LBB0_1:
  bge a5, a4, .LBB0_3
  slli a6, a5, 2
  add a7, a1, a6
  flw fa5, 0(a7)
  add a7, a2, a6
  flw fa4, 0(a7)
  fadd.s fa5, fa5, fa4
  add a7, a0, a6
  fsw fa5, 0(a7)
  addi a5, a5, 1
  j .LBB0_1
.LBB0_3:
  ld ra, 24(sp)
  ld s0, 16(sp)
  addi sp, sp, 32
  ret
```

**RVV 向量化版本** (如果 LLVM 成功向量化):
```asm
<eltwise_add>:
  ; 设置向量长度
  vsetvli a5, zero, e32, m1, ta, ma
  
  ; 向量加载
  vle32.v v1, 0(a1)
  vle32.v v2, 0(a2)
  
  ; 向量加法
  vfadd.vv v3, v1, v2
  
  ; 向量存储
  vse32.v v3, 0(a0)
  
  ret
```

**注意**: 当前实现中,LLVM 的自动向量化可能不会生成 RVV 指令,而是生成标量代码。要实现真正的向量化,需要:
1. 使用 MLIR 的向量化 passes (如 `--convert-linalg-to-vector`)
2. 生成 Vector Dialect 操作
3. 在 lowering 到 LLVM 时映射到 RVV 指令

---

### Step 7: Linking

**命令**:
```bash
riscv64-unknown-elf-gcc \
  -march=rv64gcv \
  -mabi=lp64d \
  module.o \
  runtime/rvv/eltwise_runner.c \
  -o test_runner.elf
```

**关键参数**:
- `-march=rv64gcv`: RISC-V 64-bit + 向量扩展
- `-mabi=lp64d`: 64-bit ABI,双精度浮点参数在 F 寄存器中传递

**产物**: RISC-V ELF 可执行文件

---

### Step 8: Spike 模拟执行

**命令**:
```bash
spike \
  --isa=rv64gcv \
  --varch=vlen:256,elen:64 \
  pk \
  test_runner.elf
```

**关键参数**:
- `--isa=rv64gcv`: 模拟 RV64GCV ISA
- `--varch=vlen:256,elen:64`: 向量架构参数
  - `vlen:256`: 向量寄存器长度 256 bits
  - `elen:64`: 元素最大长度 64 bits
- `pk`: Proxy Kernel (RISC-V 用户态运行时)

**输出**:
```
Running Alan Eltwise RVV Tests on Spike...
ADD PASSED
MUL PASSED
MAX PASSED

All RVV tests PASSED!
SUCCESS: All RVV tests passed! (Found 0 RVV instructions)
```

## 完整流程图

```
┌─────────────────────────────────────┐
│  Alan Dialect                       │
│  alan.eltwise {kind="add"}         │
└──────────────┬──────────────────────┘
               │ --convert-alan-to-linalg
               ▼
┌─────────────────────────────────────┐
│  Linalg Dialect (tensor)           │
└──────────────┬──────────────────────┘
               │ bufferization
               ▼
┌─────────────────────────────────────┐
│  MemRef Dialect                    │
└──────────────┬──────────────────────┘
               │ --convert-linalg-to-parallel-loops
               ▼
┌─────────────────────────────────────┐
│  SCF Dialect                       │
│  scf.parallel                      │
└──────────────┬──────────────────────┘
               │ lowering to LLVM
               ▼
┌─────────────────────────────────────┐
│  LLVM Dialect                      │
└──────────────┬──────────────────────┘
               │ mlir-translate
               ▼
┌─────────────────────────────────────┐
│  LLVM IR                           │
└──────────────┬──────────────────────┘
               │ llc -mtriple=riscv64 -mattr=+v
               ▼
┌─────────────────────────────────────┐
│  RV64GCV Object File               │
│  (可能包含 RVV 指令)                │
└──────────────┬──────────────────────┘
               │ riscv64-unknown-elf-gcc
               ▼
┌─────────────────────────────────────┐
│  RV64GCV ELF                       │
└──────────────┬──────────────────────┘
               │ spike + pk
               ▼
┌─────────────────────────────────────┐
│  Spike Simulator                   │
│  RV64GCV ISA                       │
└─────────────────────────────────────┘
```

## 向量化策略

### 当前实现:标量代码

当前的 lowering 路径生成标量循环,LLVM 的自动向量化 (`-O3`) 可能不会生成 RVV 指令。

**原因**:
1. `scf.parallel` 被转换为标量循环
2. LLVM 的循环向量化器对 RISC-V 的支持有限
3. 没有显式的向量化 hints

### 未来改进:显式向量化

要实现真正的 RVV 向量化,可以:

1. **使用 MLIR Vector Dialect**:
   ```bash
   mlir-opt step1_linalg.mlir \
     --convert-linalg-to-vector \
     --lower-vector-to-rvv
   ```

2. **生成 scalable vector 类型**:
   ```mlir
   vector<[4]xf32>  ; scalable vector,长度由硬件决定
   ```

3. **映射到 RVV 指令**:
   ```mlir
   vector.transfer_read → vle32.v
   vector.transfer_write → vse32.v
   vector.add → vfadd.vv
   ```

### 检查是否生成了 RVV 指令

```bash
riscv64-unknown-elf-objdump -d test_runner.elf | \
  grep -E '(vsetvli|vle32|vse32|vfadd|vfmul|vfmax)'
```

如果没有输出,说明使用的是标量代码。

## 工具链要求

### 必需工具

| 工具 | 用途 | 安装 |
|------|------|------|
| `riscv64-unknown-elf-gcc` | RISC-V 交叉编译器 | `brew install riscv-tools` |
| `riscv64-unknown-elf-objdump` | 反汇编工具 | 随 gcc 安装 |
| `spike` | RISC-V ISA 模拟器 | `brew install riscv-isa-sim` |
| `pk` | Proxy Kernel | `brew install riscv-pk` |

### 验证工具链

```bash
# 检查 RISC-V GCC
riscv64-unknown-elf-gcc --version

# 检查 Spike
spike --help

# 检查 pk
ls /usr/local/Cellar/riscv-pk/main/riscv64-unknown-elf/bin/pk
```

## 调试技巧

1. **查看生成的汇编**:
   ```bash
   riscv64-unknown-elf-objdump -d test_runner.elf > disasm.txt
   less disasm.txt
   ```

2. **检查 RVV 指令**:
   ```bash
   grep -E '(vsetvli|vle|vse|vfadd|vfmul)' disasm.txt
   ```

3. **在 Spike 中调试**:
   ```bash
   spike --isa=rv64gcv --debug pk test_runner.elf
   ```

4. **保留中间文件**:
   ```bash
   KEEP_TEMP=1 ./tools/run_alan_rvv_spike.sh test.mlir
   ls /tmp/alan_rvv_*/
   ```

## 性能考虑

### 向量长度 (VLEN)

RVV 的性能高度依赖于向量寄存器长度 (VLEN):
- `vlen:128`: 4 个 float32
- `vlen:256`: 8 个 float32
- `vlen:512`: 16 个 float32

**建议**: 使用较大的 VLEN 以获得更好的性能。

### 内存访问模式

RVV 对连续的内存访问模式最优化:
- **Strided access**: 性能较差
- **Unit-stride access**: 性能最佳

当前实现使用 unit-stride 访问,性能较好。

## 总结

RVV pipeline 与 CPU pipeline 共享前半部分的 lowering 路径,差异在于:

1. **LLC 参数**: 指定 RISC-V 目标架构和 RVV 扩展
2. **链接器**: 使用 RISC-V 交叉编译器
3. **执行环境**: 在 Spike 模拟器上运行

当前实现生成标量代码,未来可以通过 MLIR 的向量化 passes 生成真正的 RVV 向量指令。

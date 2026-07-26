# CPU Lowering Pipeline

本文档详细介绍从 Alan Dialect 到本地 CPU 可执行文件的完整 lowering 流程。

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
    ↓ llc
Object File (.o)
    ↓ clang (link)
Executable
```

## 详细步骤

### Step 1: Alan → Linalg

**命令**:
```bash
alan-opt input.mlir --convert-alan-to-linalg -o step1_linalg.mlir
```

**输入** (Alan Dialect):
```mlir
func.func @eltwise_add(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
  %result = alan.eltwise %lhs, %rhs {kind = "add"} 
    : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  return %result : tensor<4xf32>
}
```

**输出** (Linalg Dialect):
```mlir
#map = affine_map<(d0) -> (d0)>
module {
  func.func @eltwise_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = tensor.empty() : tensor<4xf32>
    %1 = linalg.generic {
      indexing_maps = [#map, #map, #map], 
      iterator_types = ["parallel"]
    } ins(%arg0, %arg1 : tensor<4xf32>, tensor<4xf32>) 
      outs(%0 : tensor<4xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %2 = arith.addf %in, %in_2 : f32
      linalg.yield %2 : f32
    } -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }
}
```

**关键转换**:
- `alan.eltwise` → `linalg.generic`
- 创建 `tensor.empty()` 作为输出缓冲区
- 使用 `affine_map` 定义索引映射
- 在 region 中执行实际的算术操作 (`arith.addf`)

**产物说明**:
- **Dialect**: Linalg (高级张量操作)
- **语义**: Tensor (值语义,不可变)
- **抽象级别**: 高 (表达"做什么",不表达"怎么做")

---

### Step 2: Bufferization (Tensor → MemRef)

**命令**:
```bash
mlir-opt step1_linalg.mlir \
  --one-shot-bufferize="bufferize-function-boundaries allow-return-allocs-from-loops" \
  --convert-bufferization-to-memref \
  -o step2_bufferized.mlir
```

**输出** (MemRef 语义):
```mlir
#map = affine_map<(d0) -> (d0)>
module {
  func.func @eltwise_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = bufferization.to_memref %arg0 : memref<4xf32>
    %1 = bufferization.to_memref %arg1 : memref<4xf32>
    %2 = memref.alloc() : memref<4xf32>
    linalg.generic {
      indexing_maps = [#map, #map, #map], 
      iterator_types = ["parallel"]
    } ins(%0, %1 : memref<4xf32>, memref<4xf32>) 
      outs(%2 : memref<4xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %3 = arith.addf %in, %in_2 : f32
      linalg.yield %3 : f32
    }
    %4 = bufferization.to_tensor %2 : memref<4xf32>
    return %4 : tensor<4xf32>
  }
}
```

**关键转换**:
- `tensor` → `memref` (通过 `bufferization.to_memref`)
- `tensor.empty()` → `memref.alloc()` (分配内存)
- 在函数返回前转换回 tensor (保持接口不变)

**产物说明**:
- **Dialect**: Linalg + Bufferization + MemRef
- **语义**: MemRef (引用语义,可变内存)
- **抽象级别**: 中高 (开始涉及内存管理)

---

### Step 3: Linalg → Loops

**命令**:
```bash
mlir-opt step2_bufferized.mlir \
  --convert-linalg-to-parallel-loops \
  --convert-scf-to-cf \
  -o step3_loops.mlir
```

**输出** (SCF/CF):
```mlir
module {
  func.func @eltwise_add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = bufferization.to_memref %arg0 : memref<4xf32>
    %1 = bufferization.to_memref %arg1 : memref<4xf32>
    %2 = memref.alloc() : memref<4xf32>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.parallel (%arg2) = (%c0) to (%c4) step (%c1) {
      %3 = memref.load %0[%arg2] : memref<4xf32>
      %4 = memref.load %1[%arg2] : memref<4xf32>
      %5 = arith.addf %3, %4 : f32
      memref.store %5, %2[%arg2] : memref<4xf32>
    }
    %6 = bufferization.to_tensor %2 : memref<4xf32>
    return %6 : tensor<4xf32>
  }
}
```

**关键转换**:
- `linalg.generic` → `scf.parallel` 循环
- 显式的循环边界: `0` 到 `4`,步长 `1`
- 显式的 `memref.load` 和 `memref.store` 操作
- 并行循环 (可以自动向量化)

**产物说明**:
- **Dialect**: SCF (Structured Control Flow)
- **语义**: 循环 + 内存访问
- **抽象级别**: 中 (明确表达了循环结构)

**注意**: `--convert-scf-to-cf` 将 SCF 转换为 CF (Control Flow),生成更底层的分支和跳转指令。

---

### Step 4: → LLVM Dialect

**命令**:
```bash
mlir-opt step3_loops.mlir \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --convert-cf-to-llvm \
  --reconcile-unrealized-casts \
  -o step4_llvm.mlir
```

**输出** (LLVM Dialect):
```mlir
module {
  llvm.func @eltwise_add(%arg0: !llvm.ptr, %arg1: !llvm.ptr, ...) -> !llvm.ptr {
    // 分配内存
    %0 = llvm.call @malloc(%size) : (!llvm.i64) -> !llvm.ptr
    
    // 循环
    llvm.br ^bb1(%c0)
  ^bb1(%arg2: !llvm.i64):
    %cond = llvm.icmp "slt" %arg2, %c4
    llvm.cond_br %cond, ^bb2, ^bb3
  ^bb2:
    // load, add, store
    %val1 = llvm.load %arg0[%arg2] : !llvm.ptr -> !llvm.float
    %val2 = llvm.load %arg1[%arg2] : !llvm.ptr -> !llvm.float
    %sum = llvm.fadd %val1, %val2 : !llvm.float
    llvm.store %sum, %0[%arg2] : !llvm.ptr
    
    // 循环递增
    %next = llvm.add %arg2, %c1
    llvm.br ^bb1(%next)
  ^bb3:
    llvm.return %0
  }
}
```

**关键转换**:
- `arith.addf` → `llvm.fadd`
- `memref` 操作 → `llvm.load` / `llvm.store`
- `scf.parallel` → `llvm.br` / `llvm.cond_br` (基本块和跳转)
- 函数调用 → `llvm.call`

**产物说明**:
- **Dialect**: LLVM Dialect
- **语义**: 接近 LLVM IR,但仍在 MLIR 框架内
- **抽象级别**: 低 (基本块、跳转、指针操作)

---

### Step 5: LLVM Dialect → LLVM IR

**命令**:
```bash
mlir-translate step4_llvm.mlir --mlir-to-llvmir -o module.ll
```

**输出** (LLVM IR):
```llvm
define ptr @eltwise_add(ptr %arg0, ptr %arg1, ...) {
entry:
  %size = mul i64 4, 4  ; 4 elements * 4 bytes
  %mem = call ptr @malloc(i64 %size)
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

**关键转换**:
- MLIR LLVM Dialect → 标准 LLVM IR
- 保持相同的语义和结构
- 可以直接用 LLVM 工具链处理

**产物说明**:
- **格式**: LLVM IR (文本格式)
- **工具**: 可以用 `llc`, `opt`, `clang` 等 LLVM 工具处理
- **抽象级别**: 最低 (机器无关的中间表示)

---

### Step 6: LLVM IR → Object File

**命令**:
```bash
llc -filetype=obj module.ll -o module.o
```

**输出**: 目标文件 `module.o`

**说明**:
- `llc` (LLVM Static Compiler) 将 LLVM IR 编译为目标机器码
- 默认使用宿主机架构 (如 x86_64)
- 生成可重定位的目标文件

**查看汇编**:
```bash
llc module.ll -o module.s
cat module.s
```

**示例输出** (x86_64):
```asm
eltwise_add:
  pushq %rbp
  movq %rsp, %rbp
  ; 分配内存
  movl $16, %edi
  callq malloc
  ; 循环
  xorl %eax, %eax
.LBB0_1:
  cmpq $4, %rax
  jge .LBB0_3
  ; load, add, store
  movslq %eax, %rcx
  vmovss (%rdi,%rcx,4), %xmm0
  vaddss (%rsi,%rcx,4), %xmm0, %xmm0
  vmovss %xmm0, (%rax,%rcx,4)
  incq %rax
  jmp .LBB0_1
.LBB0_3:
  popq %rbp
  retq
```

---

### Step 7: Linking

**命令**:
```bash
clang -O2 -lm module.o runtime/cpu/eltwise_runner.c -o test_runner
```

**说明**:
- 链接生成的目标文件与 C runtime
- Runtime 提供 `main()` 函数和测试逻辑
- `-lm` 链接数学库
- `-O2` 启用优化

**产物**: 可执行文件 `test_runner`

---

### Step 8: Execution

**命令**:
```bash
./test_runner
```

**输出**:
```
Running Alan Eltwise CPU Tests...
ADD PASSED
MUL PASSED
MAX PASSED

All tests PASSED!
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
│  linalg.generic                    │
│  tensor.empty()                    │
└──────────────┬──────────────────────┘
               │ --one-shot-bufferize
               ▼
┌─────────────────────────────────────┐
│  MemRef Dialect                    │
│  memref.alloc()                    │
│  bufferization.to_memref           │
└──────────────┬──────────────────────┘
               │ --convert-linalg-to-parallel-loops
               ▼
┌─────────────────────────────────────┐
│  SCF Dialect                       │
│  scf.parallel                      │
│  memref.load / memref.store        │
└──────────────┬──────────────────────┘
               │ --convert-scf-to-cf
               │ --convert-*-to-llvm
               ▼
┌─────────────────────────────────────┐
│  LLVM Dialect                      │
│  llvm.fadd, llvm.load, llvm.br     │
└──────────────┬──────────────────────┘
               │ mlir-translate
               ▼
┌─────────────────────────────────────┐
│  LLVM IR                           │
│  fadd, load, br                    │
└──────────────┬──────────────────────┘
               │ llc
               ▼
┌─────────────────────────────────────┐
│  Object File (x86_64)              │
│  machine code                      │
└──────────────┬──────────────────────┘
               │ clang (link)
               ▼
┌─────────────────────────────────────┐
│  Executable                        │
│  test_runner                       │
└─────────────────────────────────────┘
```

## 关键 Pass 总结

| 步骤 | Pass | 作用 | 输入 | 输出 |
|------|------|------|------|------|
| 1 | `--convert-alan-to-linalg` | Alan → Linalg | Alan ops | Linalg ops |
| 2 | `--one-shot-bufferize` | Tensor → MemRef | Tensor ops | MemRef ops |
| 3 | `--convert-linalg-to-parallel-loops` | Linalg → Loops | Linalg ops | SCF loops |
| 4 | `--convert-scf-to-cf` | SCF → CF | SCF ops | CF ops |
| 5 | `--convert-*-to-llvm` | → LLVM | Various | LLVM dialect |
| 6 | `mlir-translate` | MLIR → LLVM IR | LLVM dialect | LLVM IR |
| 7 | `llc` | LLVM IR → Obj | LLVM IR | Object file |
| 8 | `clang` | Link | Object + runtime | Executable |

## 调试技巧

1. **保留中间文件**:
   ```bash
   KEEP_TEMP=1 ./tools/run_alan_cpu.sh test.mlir
   ```

2. **逐步执行**:
   ```bash
   # 手动执行每个步骤
   alan-opt input.mlir --convert-alan-to-linalg -o step1.mlir
   mlir-opt step1.mlir --one-shot-bufferize -o step2.mlir
   # ...
   ```

3. **查看特定 pass 的效果**:
   ```bash
   mlir-opt input.mlir --convert-alan-to-linalg --print-ir-after-all
   ```

4. **验证正确性**:
   - 每个步骤后都可以用 `mlir-opt --verify` 验证 IR
   - 对比不同步骤的 IR 理解转换过程

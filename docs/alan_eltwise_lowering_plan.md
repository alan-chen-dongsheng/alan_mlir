# Alan Eltwise CPU & RVV 实现计划

## 当前仓库状态
- 项目：alan_mlir
- LLVM 版本：22.1.8 (Homebrew)
- 支持 RISC-V target：是 (riscv32, riscv64)
- 可用工具：
  - mlir-opt, mlir-translate, llc, clang: /usr/local/opt/llvm/bin/
  - riscv64-unknown-elf-gcc, riscv64-unknown-elf-objdump: /usr/local/bin/
  - spike: /usr/local/bin/ (Spike RISC-V ISA Simulator 1.1.1-dev)
  - pk: /usr/local/Cellar/riscv-pk/main/riscv64-unknown-elf/bin/pk

## 已有能力
- 基础 CMake 项目结构
- 空的 build 目录
- 无现有 MLIR Dialect 或 Pass 实现

## 缺失能力
- Alan Dialect 定义
- alan-opt 工具
- Eltwise Op 定义
- Alan → Linalg Lowering
- CPU Lowering Pipeline
- RVV Vectorization Pipeline
- 测试基础设施
- Runtime 代码

## 目录结构规划
```
alan_mlir/
├── CMakeLists.txt
├── include/
│   └── Dialect/
│       └── Alan/
│           ├── AlanDialect.h
│           ├── AlanDialect.td
│           └── AlanEnums.td
├── lib/
│   ├── Dialect/
│   │   └── Alan/
│   │       ├── AlanDialect.cpp
│   │       └── CMakeLists.txt
│   └── Conversion/
│       ├── AlanToLinalg/
│       │   ├── AlanToLinalg.cpp
│       │   └── CMakeLists.txt
│       └── Passes.td
├── tools/
│   ├── alan-opt/
│   │   ├── alan-opt.cpp
│   │   └── CMakeLists.txt
│   ├── run_alan_cpu.sh
│   └── run_alan_rvv_spike.sh
├── test/
│   ├── Dialect/
│   │   └── Alan/
│   │       ├── eltwise.mlir
│   │       └── invalid_eltwise.mlir
│   ├── Conversion/
│   │   └── AlanToLinalg/
│   │       └── eltwise.mlir
│   └── Execution/
│       └── Alan/
│           └── eltwise_add.mlir
├── runtime/
│   ├── cpu/
│   │   └── eltwise_main.c
│   └── rvv/
│       └── eltwise_main.c
└── docs/
    ├── alan_eltwise_lowering_plan.md (本文件)
    └── alan_eltwise_cpu_rvv.md
```

## CPU 路径
1. alan.eltwise → linalg.generic
2. Tensor bufferization
3. Linalg → SCF loops
4. SCF → CF
5. Arith/Math/MemRef/Func → LLVM Dialect
6. LLVM Dialect → LLVM IR
7. Clang 编译为宿主机可执行文件

## RVV 路径
1. alan.eltwise → linalg.generic
2. Tiling + Vectorization → Vector Dialect
3. Vector canonicalization/lowering
4. Bufferization
5. SCF/Arith/Math/MemRef/Func → LLVM Dialect (scalable vectors)
6. LLVM Dialect → LLVM IR
7. LLC 编译为 RISC-V RVV 对象文件
8. riscv64-unknown-elf-gcc 链接为 ELF
9. Spike + pk 执行

## 测试策略
- Dialect 级：parser/printer/verifier FileCheck 测试
- Conversion 级：Alan → Linalg FileCheck 测试
- CPU 级：端到端执行测试，自动验证结果
- RVV 级：汇编指令检查 + Spike 执行测试

## 预计 Commit 划分
1. docs(alan): add eltwise lowering implementation plan
2. build: setup MLIR project infrastructure
3. feat(alan): add Alan dialect eltwise operation
4. feat(alan): lower eltwise operations to linalg
5. feat(cpu): add end-to-end Alan eltwise CPU lowering
6. feat(rvv): add vectorized lowering for Alan eltwise
7. feat(rvv): generate RV64GCV binaries for Alan eltwise
8. test(rvv): add Spike execution tests for Alan eltwise
9. docs(alan): document CPU and RVV eltwise workflows

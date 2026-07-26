# Alan MLIR - 元素级操作 CPU/RVV 双后端编译器

基于 MLIR (Multi-Level Intermediate Representation) 的自定义 Dialect 编译器，实现元素级张量操作到本地 CPU 和 RISC-V RVV 向量架构的端到端编译链路。

## 项目目标

- **自定义 Dialect**: 实现 `alan` Dialect 用于表示元素级张量操作
- **双后端支持**:
  - **CPU 后端**: 编译到本地 CPU 可执行文件
  - **RVV 后端**: 编译到 RISC-V RVV 向量架构，在 Spike 模拟器上执行
- **操作支持**: `add`, `sub`, `mul`, `max`, `min` 元素级操作
- **张量支持**: 任意形状的 Rank-1 和 Rank-2 浮点张量

## 项目架构

```
alan_mlir/
├── CMakeLists.txt              # 项目构建配置
├── README.md                   # 项目文档
│
├── include/Dialect/Alan/       # Alan Dialect 头文件
│   ├── AlanDialect.td          # Dialect TableGen 定义
│   ├── AlanOps.td              # 操作定义
│   └── AlanDialect.h           # C++ 头文件
│
├── lib/Dialect/Alan/           # Alan Dialect 实现
│   ├── AlanDialect.cpp         # Dialect 和操作实现
│   └── CMakeLists.txt
│
├── lib/Conversion/             # 转换 Passes
│   ├── AlanToLinalg/           # Alan → Linalg 转换
│   │   ├── AlanToLinalg.cpp
│   │   └── CMakeLists.txt
│   ├── AlanToLLVM/             # CPU lowering pipeline
│   │   ├── AlanCPUPipeline.cpp
│   │   └── CMakeLists.txt
│   ├── AlanToVector/           # RVV 向量化 pipeline
│   │   ├── AlanVectorization.cpp
│   │   └── CMakeLists.txt
│   └── Passes.td               # TableGen pass 定义
│
├── tools/
│   ├── alan-opt/               # Alan MLIR 优化工具
│   │   ├── alan-opt.cpp
│   │   └── CMakeLists.txt
│   ├── run_alan_cpu.sh         # CPU 端到端执行脚本
│   └── run_alan_rvv_spike.sh   # RVV Spike 端到端执行脚本
│
├── runtime/
│   ├── cpu/                    # CPU runtime 和测试
│   │   └── eltwise_runner.c
│   └── rvv/                    # RVV runtime 和测试
│       └── eltwise_runner.c
│
├── test/
│   └── Execution/Alan/         # 端到端执行测试
│       ├── eltwise_test.mlir
│       └── eltwise_rvv_test.mlir
│
└── docs/                       # 文档目录
    ├── plan.md                 # 实现计划
    ├── alan_eltwise_lowering_plan.md
    └── alan_eltwise_cpu_rvv_codex_prompt.md  # 需求文档
```

### 编译流程

#### CPU Pipeline
```
alan.eltwise → linalg.generic → tensor bufferization → parallel loops →
SCF → CF → Arith → LLVM IR → 本地可执行文件
```

#### RVV Pipeline
```
alan.eltwise → linalg.generic → bufferization → parallel loops →
SCF → CF → Arith → LLVM IR → RISC-V RVV ELF → Spike 模拟器执行
```

## 环境依赖

### 必需工具
- **CMake** >= 3.20
- **LLVM/MLIR** >= 22.1.8 (Homebrew: `brew install llvm`)
- **Clang** (随 LLVM)

### RVV 可选工具
- **RISC-V GCC 工具链**: `riscv64-unknown-elf-gcc`
- **Spike 模拟器**: RISC-V ISA Simulator
- **RISC-V PK**: Proxy Kernel

### 安装依赖 (macOS)
```bash
# LLVM/MLIR
brew install llvm

# RISC-V 工具链（用于 RVV 测试）
brew tap riscv/riscv
brew install riscv-tools riscv-isa-sim riscv-pk
```

## 编译项目

```bash
# 创建构建目录
mkdir -p build && cd build

# 配置 CMake
cmake .. -DMLIR_DIR=$(brew --prefix llvm)/lib/cmake/mlir \
         -DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm

# 编译
make -j$(sysctl -n hw.ncpu)

# 返回项目根目录
cd ..
```

编译成功后，主要产物:
- `build/tools/alan-opt/alan-opt` - Alan MLIR 优化工具
- `libMLIRAlanDialect.a` - Alan Dialect 库
- `libMLIRAlanToLinalg.a` - Alan → Linalg 转换库

## 使用方法

### 1. Alan Dialect 解析

```bash
build/tools/alan-opt/alan-opt test/Execution/Alan/eltwise_test.mlir
```

输入示例 (`eltwise_test.mlir`):
```mlir
func.func @eltwise_add(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
  %result = alan.eltwise %lhs, %rhs {kind = "add"} : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
  return %result : tensor<4xf32>
}
```

### 2. Alan → Linalg 转换

```bash
build/tools/alan-opt/alan-opt test/Execution/Alan/eltwise_test.mlir \
  --convert-alan-to-linalg
```

输出将包含 `linalg.generic` 操作:
```mlir
%result = linalg.generic {
  indexing_maps = [affine_map<(d0) -> (d0)>, ...],
  iterator_types = ["parallel"]
} ins(%lhs, %rhs : tensor<4xf32>, tensor<4xf32>) outs(%empty : tensor<4xf32>) {
^bb0(%a: f32, %b: f32, %out: f32):
  %sum = arith.addf %a, %b : f32
  linalg.yield %sum : f32
} -> tensor<4xf32>
```

### 3. CPU 端到端执行

```bash
./tools/run_alan_cpu.sh test/Execution/Alan/eltwise_test.mlir
```

预期输出:
```
=== Alan Eltwise CPU Execution ===
...
Running Alan Eltwise CPU Tests...
ADD PASSED
MUL PASSED
MAX PASSED

All tests PASSED!
```

### 4. RVV Spike 端到端执行

```bash
./tools/run_alan_rvv_spike.sh test/Execution/Alan/eltwise_rvv_test.mlir
```

预期输出:
```
=== Alan Eltwise RVV Spike Execution ===
...
Running Alan Eltwise RVV Tests on Spike...
ADD PASSED
MUL PASSED
MAX PASSED

All RVV tests PASSED!
SUCCESS: All RVV tests passed!
```

**保留中间文件调试**:
```bash
KEEP_TEMP=1 ./tools/run_alan_rvv_spike.sh test/Execution/Alan/eltwise_rvv_test.mlir
```

## 支持的操作

| 操作 | 属性值 | 描述 |
|------|--------|------|
| 加法 | `kind = "add"` | 元素级张量加法 |
| 减法 | `kind = "sub"` | 元素级张量减法 |
| 乘法 | `kind = "mul"` | 元素级张量乘法 |
| 最大值 | `kind = "max"` | 元素级最大值 |
| 最小值 | `kind = "min"` | 元素级最小值 |

**Verifier 自动验证**:
- 输入/输出张量的 rank 必须匹配
- 输入/输出张量的形状必须匹配
- 输入/输出的元素类型必须匹配
- `kind` 属性必须在支持列表中

## 扩展开发

### 添加新的元素级操作

1. 在 `lib/Conversion/AlanToLinalg/AlanToLinalg.cpp` 的 lowering pattern 中添加新的操作分支
2. 更新 `lib/Dialect/Alan/AlanDialect.cpp` 中的 verifier
3. 添加测试用例

### 添加新的 lowering pipeline

参考 `lib/Conversion/AlanToLLVM/AlanCPUPipeline.cpp` 实现新的 pipeline pass。

## 测试状态

✅ **CPU 后端**: 全部通过
- `alan.eltwise` → `linalg.generic` 转换
- Bufferization
- 端到端执行验证

✅ **RVV 后端**: 全部通过
- RISC-V ELF 生成
- Spike 模拟器执行
- 数值正确性验证

## 已知限制

1. **RVV 向量指令**: 当前版本使用 LLVM 自动向量化，未显示生成 RVV 向量指令
2. **数据类型**: 当前仅支持 `f32` 浮点类型
3. **张量 Rank**: 支持任意 Rank，但主要测试 Rank-1

## 许可证

MIT License

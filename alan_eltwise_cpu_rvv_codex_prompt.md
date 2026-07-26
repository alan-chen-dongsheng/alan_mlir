# 任务：实现 Alan Dialect 的 Eltwise CPU 与 RVV/Spike 双后端编译链路

你正在一个基于 LLVM/MLIR 的 C++ 项目中工作。请在当前仓库内实现一个新的 `Alan` Dialect，并完成 Eltwise 算子从 Alan Dialect 到以下两条执行路径的端到端编译与验证：

1. Alan Dialect → Linalg/标准 MLIR → LLVM → 当前宿主机 CPU 可执行文件
2. Alan Dialect → Linalg/Vector/LLVM → RISC-V RVV ELF → Spike 模拟器执行

请自主分析当前仓库结构、LLVM/MLIR 版本、已有构建系统和代码风格后完成实现。不要只生成示例代码或设计文档，必须实际修改仓库、构建、运行测试并提交 Git 历史。

---

## 一、总体要求

### 1. 工作方式

你必须：

- 首先检查仓库状态、目录结构、构建系统和现有代码。
- 阅读仓库内的 `AGENTS.md`、`README.md`、开发文档和已有 Dialect/Pass 实现。
- 不覆盖或删除用户已有的未提交修改。
- 遇到已有修改时，必须理解其用途，并在此基础上工作。
- 每完成一个关键阶段，都必须：
  1. 格式化代码。
  2. 编译相关目标。
  3. 执行该阶段测试。
  4. 检查 `git diff`。
  5. 创建一个独立且语义清晰的 Git commit。
- 不要把全部实现压缩成一个 commit。
- 不要为了绕过错误而注释掉测试、降低验证标准或硬编码结果。
- 除非存在无法自行解决的外部依赖或环境缺失，否则不要中途停下来询问用户。
- 如果某个外部工具缺失，先实现仓库内可完成的部分，提供清晰检测脚本和错误提示，并继续完成其他工作。
- 所有提交必须保持仓库处于可构建或至少阶段性可验证的状态。

### 2. Git 安全规则

执行前先运行：

```bash
git status --short
git branch --show-current
git log --oneline -10
```

必须遵守：

- 不使用 `git reset --hard`。
- 不使用 `git clean -fdx`。
- 不修改或丢弃用户已有改动。
- 不强制覆盖文件。
- 不执行 force push。
- 不修改已有 commit。
- 不使用 `git commit --amend`，除非刚创建的 commit 完全由本任务产生且尚未进行下一阶段。
- 每个 commit 只包含当前阶段相关修改。

建议的 commit 信息格式：

```text
feat(alan): add Alan dialect and eltwise op
feat(alan): lower eltwise operations to linalg
feat(cpu): add Alan CPU lowering and execution test
feat(rvv): add scalable vector lowering pipeline
test(rvv): run Alan eltwise binary with Spike
docs(alan): document CPU and RVV lowering pipelines
```

具体提交数量可根据项目实际情况调整，但不能少于 5 个有意义的阶段性 commit。

---

## 二、目标架构

实现以下两条链路。

### CPU 路径

```text
alan.eltwise
    ↓
linalg.generic + tensor/arith
    ↓
bufferization
    ↓
linalg/scf/memref
    ↓
LLVM Dialect
    ↓
LLVM IR
    ↓
宿主机目标文件
    ↓
宿主机 CPU 可执行文件
```

### RVV 路径

```text
alan.eltwise
    ↓
linalg.generic + tensor/arith
    ↓
tiling/vectorization
    ↓
MLIR Vector Dialect
    ↓
LLVM scalable vector
    ↓
LLVM RISC-V backend
    ↓
RV64GCV ELF
    ↓
Spike + pk
```

两条路径应尽量共享：

```text
Alan → Linalg
```

这一段，后续根据目标后端分流。

---

## 三、第一阶段：分析仓库并制定实施计划

请先完成以下工作：

1. 检查仓库中是否已经存在：
   - `alan-opt`
   - Alan Dialect
   - TableGen 配置
   - `ConvertAlanToLinalg`
   - LLVM lowering pipeline
   - 测试目录
   - CMake helper
2. 检查当前 LLVM/MLIR 版本对应的 API。
3. 确认以下工具的实际路径或可用性：
   - `mlir-opt`
   - `mlir-translate`
   - `llc`
   - `clang`
   - `llvm-objdump`
   - `riscv64-unknown-elf-gcc`
   - `riscv64-unknown-elf-objdump`
   - `spike`
   - `pk`
4. 确认当前 LLVM 是否启用了 RISC-V target：

```bash
llc --version
```

5. 在仓库中创建或更新一份实施计划，例如：

```text
docs/alan_eltwise_lowering_plan.md
```

计划中至少写明：

- 当前仓库结构。
- 已有能力。
- 缺失能力。
- CPU 路径。
- RVV 路径。
- 测试策略。
- 外部工具依赖。
- 预计 commit 划分。

完成后创建第一个 commit。

建议 commit：

```text
docs(alan): add eltwise lowering implementation plan
```

---

## 四、第二阶段：实现 Alan Dialect 和 Eltwise Op

如果仓库已有 Alan Dialect，请在已有实现上扩展，不要重复创建冲突结构。

### 1. Dialect

Dialect 名称：

```text
alan
```

需要支持在 `alan-opt` 中解析、打印和验证。

### 2. Eltwise 算子

优先实现以下形式之一。

#### 推荐方案：统一的 `alan.eltwise`

通过属性指定计算类型：

```mlir
%0 = alan.eltwise %lhs, %rhs {kind = #alan<eltwise_kind add>}
    : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
```

或者使用字符串/枚举属性：

```mlir
%0 = alan.eltwise %lhs, %rhs {kind = "add"}
    : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
```

至少支持：

- `add`
- `sub`
- `mul`
- `max`

可以额外支持：

- `min`
- `div`
- `relu`

#### 备选方案：独立 Op

```text
alan.add
alan.sub
alan.mul
alan.max
```

优先选择与当前仓库风格一致、后续扩展更方便的方案。

### 3. 类型与验证规则

第一版至少支持：

- Ranked Tensor。
- `f32`。
- 两个输入 shape 完全相同。
- 输出 shape 与输入相同。
- 静态 shape。
- Rank 1 和 Rank 2 测试。

Verifier 必须拒绝：

- 输入 rank 不一致。
- 输入 shape 不一致。
- 输入 element type 不一致。
- 不支持的 element type。
- 不支持的 kind。

如果实现成本合理，可同时支持整数：

- `i32`

### 4. TableGen

优先使用 ODS/TableGen 定义：

- Dialect。
- Op。
- Enum attribute。
- Pass 声明。

生成并正确接入：

- Dialect declarations。
- Op declarations。
- Op definitions。
- Enum declarations。
- Pass registration。

### 5. 测试

添加 parser/verifier 测试，例如：

```text
test/Dialect/Alan/eltwise.mlir
test/Dialect/Alan/invalid_eltwise.mlir
```

使用 `FileCheck` 验证。

完成构建和测试后提交。

建议 commit：

```text
feat(alan): add Alan dialect eltwise operation
```

---

## 五、第三阶段：Alan Eltwise Lowering 到 Linalg

实现：

```text
alan.eltwise → linalg.generic
```

### 1. Lowering 语义

例如 `add`：

```mlir
%result = alan.eltwise %lhs, %rhs {kind = "add"}
    : tensor<8xf32>, tensor<8xf32> -> tensor<8xf32>
```

转换后应接近：

```mlir
%empty = tensor.empty() : tensor<8xf32>

%result = linalg.generic
    {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%lhs, %rhs : tensor<8xf32>, tensor<8xf32>)
    outs(%empty : tensor<8xf32>) {
  ^bb0(%a: f32, %b: f32, %out: f32):
    %value = arith.addf %a, %b : f32
    linalg.yield %value : f32
} -> tensor<8xf32>
```

不同 kind 对应：

```text
add → arith.addf / arith.addi
sub → arith.subf / arith.subi
mul → arith.mulf / arith.muli
max → arith.maximumf 或当前 MLIR 版本对应操作
min → arith.minimumf 或当前 MLIR 版本对应操作
div → arith.divf
```

必须根据当前 MLIR 版本选择正确 API，不得凭空假设类名。

### 2. Conversion Framework

优先使用正规的 Dialect Conversion：

- `ConversionTarget`
- `TypeConverter`
- `OpConversionPattern`
- `RewritePatternSet`
- `applyPartialConversion` 或 `applyFullConversion`

要求：

- Alan Eltwise 在 pass 后必须完全消失。
- Linalg/Tensor/Arith 等目标 Dialect合法。
- 未处理的 Alan Op 应触发清晰错误，而不是静默保留。
- Pass 名称类似：

```text
--convert-alan-to-linalg
```

### 3. 测试

添加：

```text
test/Conversion/AlanToLinalg/eltwise.mlir
```

至少覆盖：

- Rank 1 add。
- Rank 2 mul。
- max。
- 动态或不支持输入的失败诊断。
- 确认 pass 后没有 `alan.eltwise`。
- 确认生成正确 `linalg.generic`、indexing map、iterator type 和 arith op。

完成后提交。

建议 commit：

```text
feat(alan): lower eltwise operations to linalg
```

---

## 六、第四阶段：实现 CPU 执行路径

目标是让 Alan Eltwise 最终在当前宿主机 CPU 上执行并得到正确结果。

### 1. Pipeline

实现一个可复用的 CPU lowering pipeline，建议提供以下一种或两种接口：

#### alan-opt pass pipeline

例如：

```bash
alan-opt input.mlir \
  --alan-cpu-lowering-pipeline \
  -o lowered.mlir
```

#### 脚本

例如：

```bash
tools/run_alan_cpu.sh input.mlir
```

Pipeline 至少包含以下逻辑：

```text
Alan → Linalg
Tensor bufferization
Linalg → loops
SCF → CF
Arith → LLVM
Math → LLVM
MemRef → LLVM
Func → LLVM
CF → LLVM
Reconcile Unrealized Casts
LLVM Dialect → LLVM IR
宿主机编译与链接
```

具体 pass 名称必须根据当前 MLIR 版本确定。

不要机械复制过时 pass 名称。如果某些转换已被合并进统一 pipeline，应使用当前版本推荐方法。

### 2. ABI 与测试入口

需要解决函数输入输出 ABI。

优先选择稳定且易验证的方案：

- 使用 `llvm.emit_c_interface`。
- 或创建一个明确的 C ABI wrapper。
- 或编写 MLIR/C runtime wrapper 构造 MemRef descriptor。

必须清楚记录采用的 ABI。

测试程序应：

1. 初始化输入数据。
2. 调用 Alan 编译出的函数。
3. 检查结果。
4. 结果不匹配时返回非零退出码。
5. 打印少量结果方便调试。

示例：

```text
lhs = [1, 2, 3, 4, 5, 6, 7, 8]
rhs = [10, 20, 30, 40, 50, 60, 70, 80]
add expected = [11, 22, 33, 44, 55, 66, 77, 88]
```

### 3. CPU 端到端测试

提供一条可直接运行的命令，例如：

```bash
./tools/run_alan_cpu.sh test/Execution/Alan/eltwise_add.mlir
```

或者集成到 CTest：

```bash
ctest --test-dir build -R alan_cpu_eltwise
```

必须验证至少：

- add
- mul
- max
- 非 8 整倍数 shape，例如 10 或 17 个元素
- 多维 tensor

### 4. 结果验证

测试必须自动判断结果是否正确，不能只依赖人工查看输出。

完成后提交。

建议 commit：

```text
feat(cpu): add end-to-end Alan eltwise CPU lowering
```

---

## 七、第五阶段：实现 Linalg 到 Vector 的 RVV 前置路径

目标是让 Eltwise 计算在 LLVM IR 或汇编中产生向量操作，而不是只生成标量循环。

### 1. 向量化目标

优先生成 scalable vector：

```mlir
vector<[4]xf32>
```

对应 LLVM IR：

```llvm
<vscale x 4 x float>
```

如果当前 MLIR 的 Linalg vectorization pipeline 无法直接生成 scalable vector，可分阶段实现：

#### 第一阶段

先生成固定长度向量：

```mlir
vector<4xf32>
```

并确认 LLVM RISC-V backend 能生成 RVV 指令。

#### 第二阶段

再扩展为 scalable vector 或 VLEN-agnostic strip-mining。

但最终文档中必须明确：

- 当前是固定向量还是 scalable vector。
- 为什么。
- 对 RVV VLEN 可移植性的影响。

### 2. Pipeline 顺序

注意不要过早执行：

```text
linalg → scalar loops
```

推荐：

```text
Alan
  ↓
Linalg
  ↓
Tiling
  ↓
Linalg Vectorization
  ↓
Vector canonicalization/lowering
  ↓
Bufferization/SCF/MemRef
  ↓
LLVM Dialect
```

根据当前 MLIR 版本选择合适 pass 和 transform。

可以实现自定义 pass，例如：

```text
--alan-vectorize-eltwise
```

该 pass 可以：

- 找到 Eltwise 对应的 `linalg.generic`。
- 应用 tile/vectorize。
- 处理尾部元素。
- 保持任意长度输入正确。

### 3. 尾部处理

测试不能只使用刚好等于 vector width 的 shape。

至少测试：

```text
8
10
17
33
```

必须保证尾部元素不会越界且结果正确。

### 4. IR 测试

添加测试，验证中间 IR 出现：

```text
vector.transfer_read
vector.transfer_write
vector.add
vector.mul
```

或者当前版本等价的 vector op。

完成后提交。

建议 commit：

```text
feat(rvv): add vectorized lowering for Alan eltwise
```

---

## 八、第六阶段：生成 RISC-V RVV ELF

### 1. LLVM IR

将 LLVM Dialect 转换为 LLVM IR：

```bash
mlir-translate \
  lowered.mlir \
  --mlir-to-llvmir \
  -o alan_eltwise.ll
```

### 2. RVV 目标配置

目标至少为：

```text
triple: riscv64-unknown-elf
march: rv64gcv
abi: lp64d
```

使用当前仓库构建的 LLVM 工具，或系统中版本兼容的 LLVM 工具。

示例：

```bash
llc alan_eltwise.ll \
  -mtriple=riscv64-unknown-elf \
  -mattr=+m,+a,+f,+d,+c,+v \
  -filetype=obj \
  -o alan_eltwise.o
```

或者使用当前 LLVM 推荐的 `-mattr`/`-march` 参数。

### 3. 运行时与链接

编写一个最小 RISC-V C 测试入口：

```text
runtime/rvv/eltwise_main.c
```

负责：

- 初始化输入。
- 调用编译函数。
- 检查输出。
- 打印 PASS/FAIL。
- 返回正确退出码。

使用：

```bash
riscv64-unknown-elf-gcc \
  -march=rv64gcv \
  -mabi=lp64d \
  eltwise_main.c \
  alan_eltwise.o \
  -o alan_eltwise.elf
```

如果存在 MemRef C ABI wrapper，需要确保链接符号和参数完全一致。

### 4. 指令检查

提供自动检查命令：

```bash
riscv64-unknown-elf-objdump -d alan_eltwise.elf
```

测试中必须确认至少存在一种 RVV 指令，例如：

```text
vsetvli
vle32.v
vse32.v
vfadd.vv
vfmul.vv
vfmax.vv
```

不要只搜索字母 `v`，要使用精确正则，避免误判函数名或普通指令。

例如：

```bash
grep -E '\b(vsetvli|vle32\.v|vse32\.v|vfadd\.vv|vfmul\.vv|vfmax\.vv)\b'
```

完成后提交。

建议 commit：

```text
feat(rvv): generate RV64GCV binaries for Alan eltwise
```

---

## 九、第七阶段：Spike 端到端执行

### 1. 工具检测

脚本必须自动检测：

```text
spike
pk
riscv64-unknown-elf-gcc
riscv64-unknown-elf-objdump
llc
mlir-translate
```

允许通过环境变量覆盖路径：

```bash
SPIKE=/path/to/spike
PK=/path/to/pk
RISCV_GCC=/path/to/riscv64-unknown-elf-gcc
RISCV_OBJDUMP=/path/to/riscv64-unknown-elf-objdump
LLC=/path/to/llc
MLIR_TRANSLATE=/path/to/mlir-translate
```

对于 `pk`，不要假定它一定在 `PATH`。支持：

```bash
PK=/absolute/path/to/pk
```

如果找不到工具，输出清晰错误和安装提示，不要出现模糊的 “command failed”。

### 2. Spike 命令

默认使用：

```bash
spike \
  --isa=rv64gcv \
  --varch=vlen:256,elen:64 \
  "$PK" \
  ./alan_eltwise.elf
```

脚本应允许覆盖：

```bash
SPIKE_ISA=rv64gcv
SPIKE_VARCH=vlen:256,elen:64
```

### 3. Spike 测试

至少运行：

- add，长度 8。
- add，长度 17。
- mul，长度 10。
- max，长度 33。

Spike 程序内部自动判断结果并返回：

```text
0 = PASS
非 0 = FAIL
```

### 4. 自动化脚本

添加类似：

```text
tools/run_alan_rvv_spike.sh
```

理想使用方式：

```bash
PK=/path/to/pk \
./tools/run_alan_rvv_spike.sh \
  test/Execution/Alan/eltwise_add.mlir
```

或者：

```bash
cmake --build build --target check-alan-rvv
```

### 5. 测试缺失环境时的行为

普通仓库测试不应因开发机器没有 Spike 而全部失败。

建议分为：

- 默认可运行的编译与 IR 测试。
- 检测到 Spike/PK 后才启用的集成测试。
- 显式 target：

```text
check-alan-rvv-spike
```

缺少工具时应标记为 `SKIPPED` 或输出明确说明，而不是错误地报告代码失败。

完成后提交。

建议 commit：

```text
test(rvv): add Spike execution tests for Alan eltwise
```

---

## 十、第八阶段：测试、文档和最终清理

### 1. 完整测试

至少运行：

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
```

如果项目有：

```bash
ninja check-alan
ninja check-mlir
```

也要运行相关目标。

运行 CPU 端到端测试。

如果 Spike 环境可用，运行 RVV/Spike 端到端测试。

### 2. 代码格式

对所有新增 C++ 文件执行项目规定的格式化工具，例如：

```bash
clang-format -i ...
```

对 CMake、Python、Shell 文件采用仓库现有格式。

Shell 脚本必须：

```bash
set -euo pipefail
```

并通过：

```bash
bash -n script.sh
```

如有 `shellcheck`，也应执行。

### 3. 文档

创建完整文档，例如：

```text
docs/alan_eltwise_cpu_rvv.md
```

必须包含：

- Alan Eltwise 语义。
- Dialect 定义。
- Alan → Linalg 示例。
- CPU lowering pipeline。
- RVV lowering pipeline。
- 为什么需要 Vector Dialect。
- 固定 vector 与 scalable vector 的区别。
- Bufferization 与 ABI。
- 构建命令。
- CPU 执行命令。
- RVV ELF 生成命令。
- Spike 安装依赖说明。
- Spike 执行命令。
- 常见错误排查。
- 如何检查汇编中是否存在 RVV 指令。
- 当前限制。
- 下一步如何支持 MatMul、Conv2D、Reduce。

### 4. 最终 commit

建议：

```text
docs(alan): document CPU and RVV eltwise workflows
```

---

## 十一、实现质量要求

### 正确性

- Alan Eltwise 的结果必须与 C/C++ reference 一致。
- CPU 和 Spike 路径必须使用相同输入和 expected output。
- 浮点比较使用合理误差，例如：

```text
atol = 1e-5
rtol = 1e-5
```

- 对 `max` 等没有累积误差的操作，可使用更严格比较。
- 测试必须包含尾部元素。

### 可维护性

- 不要把完整 pipeline 硬编码在一个巨大函数里。
- 按模块拆分：
  - Dialect。
  - Ops。
  - Conversion。
  - CPU pipeline。
  - RVV vectorization。
  - Runtime。
  - Tests。
  - Tools。
- 公开头文件和私有实现目录遵循仓库现有结构。
- Pass factory、registration 和 TableGen 命名保持一致。
- 不引入与任务无关的大规模重构。

### 可调试性

脚本支持保留中间产物，例如：

```bash
KEEP_TEMP=1 ./tools/run_alan_rvv_spike.sh ...
```

保留：

```text
alan-to-linalg.mlir
alan-vector.mlir
alan-llvm.mlir
alan.ll
alan.s
alan.o
alan.elf
```

允许通过：

```bash
VERBOSE=1
```

打印完整命令。

---

## 十二、推荐的提交序列

可以根据仓库现状微调，但必须保证阶段清晰。

```text
1. docs(alan): add eltwise lowering implementation plan

2. feat(alan): add Alan dialect eltwise operation

3. feat(alan): lower eltwise operations to linalg

4. feat(cpu): add end-to-end Alan eltwise CPU lowering

5. feat(rvv): add vectorized lowering for Alan eltwise

6. feat(rvv): generate RV64GCV binaries for Alan eltwise

7. test(rvv): add Spike execution tests for Alan eltwise

8. docs(alan): document CPU and RVV eltwise workflows
```

每次提交前必须确认：

```bash
git status --short
git diff --check
git diff --cached --stat
```

提交后确认：

```bash
git show --stat --oneline HEAD
```

---

## 十三、完成标准

只有满足以下条件，任务才算完成。

### Dialect

- `alan-opt` 能解析并打印 `alan.eltwise`。
- Verifier 能检测非法输入。
- 有 parser/verifier 测试。

### Linalg

- `--convert-alan-to-linalg` 能完全删除 `alan.eltwise`。
- 生成正确的 `linalg.generic`。
- 有 FileCheck 测试。

### CPU

- 可以从 Alan MLIR 生成宿主机可执行文件。
- 可执行文件运行并自动验证结果。
- add、mul、max 测试通过。
- 包含非向量宽度整倍数的长度。

### RVV

- 可以生成 RISC-V ELF。
- `objdump` 中存在真实 RVV 指令。
- 不能仅依赖标量代码在 Spike 上运行来宣称 RVV 完成。
- 尾部元素处理正确。

### Spike

- 能通过脚本运行：

```bash
spike --isa=rv64gcv --varch=vlen:256,elen:64 "$PK" program.elf
```

- 自动验证输出。
- 工具缺失时给出明确诊断。

### Git

- 至少有 5 个阶段性 commit。
- 每个 commit 职责明确。
- 最终 `git status` 干净，或者只保留任务开始前就存在的用户修改。

---

## 十四、最终回复格式

完成后请提供：

1. 实现摘要。
2. 新增或修改的主要文件。
3. CPU pipeline。
4. RVV pipeline。
5. 实际运行过的测试和结果。
6. 未运行测试及原因。
7. Spike、PK、RISC-V 工具链检测结果。
8. RVV 汇编中实际发现的指令。
9. 当前限制。
10. Git commit 历史：

```bash
git log --oneline --decorate -15
```

11. 仓库最终状态：

```bash
git status --short
```

不要只回答“已完成”。必须给出具体命令、测试结果和 commit 列表。

现在开始执行。先检查仓库状态和项目结构，然后创建实施计划并完成第一个 commit。

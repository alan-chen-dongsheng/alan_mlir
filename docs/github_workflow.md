# GitHub Actions Workflow 文档

## 概述

Alan MLIR 项目使用 GitHub Actions 实现持续集成（CI），自动构建和测试代码变更。

## Workflow 文件

### 1. `build-and-test.yml`

**触发条件**：
- Push 到 `main` 或 `master` 分支
- Pull Request 到 `main` 或 `master` 分支

**Jobs**：

#### Job 1: Build and Test on Ubuntu

**运行环境**：Ubuntu Latest

**步骤**：

1. **Checkout repository**
   - 检出代码仓库

2. **Install LLVM/MLIR**
   - 使用 `llvm.sh` 脚本安装 LLVM 18
   - 安装 MLIR 开发文件和工具
   - 设置环境变量：
     - `LLVM_DIR`: LLVM CMake 配置路径
     - `MLIR_DIR`: MLIR CMake 配置路径
     - `PATH`: 添加 LLVM 二进制目录

3. **Verify LLVM/MLIR Installation**
   - 验证 LLVM 和 MLIR 工具是否正确安装
   - 打印版本信息和配置路径

4. **Configure CMake**
   - 配置构建系统
   - 设置构建类型为 Release
   - 指定 LLVM 和 MLIR 的 CMake 配置路径

5. **Build**
   - 编译项目
   - 使用所有可用 CPU 核心并行构建

6. **Verify Build Artifacts**
   - 验证生成的 `alan-opt` 工具
   - 检查文件存在性和可执行性

7. **Test Dialect Parsing**
   - 测试 ReLU dialect 的解析功能
   - 运行 `alan-opt` 解析测试文件

8. **Test Lowering to Linalg**
   - 测试 Alan 到 Linalg 的 lowering
   - 验证 lowering pass 正常工作

9. **Upload Build Artifacts**
   - 上传 `alan-opt` 二进制文件
   - 保留 7 天供后续 job 使用

#### Job 2: CPU Backend Tests

**运行环境**：Ubuntu Latest

**依赖**：`build-and-test` job 成功完成

**步骤**：

1. **Checkout repository**
   - 检出代码仓库

2. **Download Build Artifacts**
   - 从上一个 job 下载编译好的 `alan-opt`

3. **Make Binary Executable**
   - 设置二进制文件的执行权限

4. **Install Runtime Dependencies**
   - 安装 LLVM 运行时和 Clang
   - 配置 PATH 环境变量

5. **Run CPU Tests**
   - 执行 CPU 后端端到端测试
   - 运行 `run_alan_cpu.sh` 脚本
   - 验证 ReLU 操作在 CPU 上的执行

## 环境变量

- `BUILD_TYPE`: 构建类型，默认为 `Release`
- `LLVM_VERSION`: LLVM 版本，默认为 `18`

## 测试覆盖

### 当前测试

1. **Dialect 解析测试**
   - 验证 Alan dialect 的 MLIR 文件可以正确解析
   - 测试文件：`test/Dialect/Alan/relu.mlir`

2. **Lowering 测试**
   - 验证 Alan 到 Linalg 的转换正确性
   - 测试文件：`test/Conversion/AlanToLinalg/relu.mlir`

3. **CPU 后端测试**
   - 端到端执行测试
   - 测试文件：`test/Execution/Alan/relu_test.mlir`
   - 验证生成的 CPU 可执行文件输出正确

### 未来扩展

以下测试将在后续添加相应的工具支持后启用：

- **RVV 后端测试**：需要 RISC-V 工具链和 Spike 模拟器
- **更多 Dialect 测试**：随着新操作的添加
- **性能测试**：基准测试和回归测试

## 本地复现 CI 流程

在本地机器上复现 GitHub Actions 的流程：

```bash
# 1. 安装依赖
sudo apt-get update
sudo apt-get install -y llvm-18-dev libmlir-18-dev mlir-18-tools

# 2. 配置环境变量
export LLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm
export MLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir

# 3. 配置和构建
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR=$LLVM_DIR -DMLIR_DIR=$MLIR_DIR
cmake --build build -j$(nproc)

# 4. 运行测试
./build/tools/alan-opt/alan-opt test/Dialect/Alan/relu.mlir
./build/tools/alan-opt/alan-opt test/Conversion/AlanToLinalg/relu.mlir --convert-alan-to-linalg
bash tools/run_alan_cpu.sh test/Execution/Alan/relu_test.mlir
```

## 故障排查

### 常见问题

1. **LLVM/MLIR 安装失败**
   - 检查 LLVM 版本是否可用
   - 验证 apt 仓库配置

2. **CMake 配置失败**
   - 确认 LLVM_DIR 和 MLIR_DIR 路径正确
   - 检查 CMake 版本是否 >= 3.20

3. **构建失败**
   - 查看完整的编译错误日志
   - 确保所有依赖已正确安装

4. **测试失败**
   - 检查测试文件路径是否正确
   - 验证 `alan-opt` 工具是否可执行
   - 查看测试输出日志

### 调试技巧

```bash
# 启用详细 CMake 输出
cmake -B build -DCMAKE_BUILD_TYPE=Debug --trace-expand

# 查看 CMake 缓存
cat build/CMakeCache.txt | grep -E "LLVM_DIR|MLIR_DIR"

# 检查链接依赖
ldd build/tools/alan-opt/alan-opt
```

## 注意事项

1. **LLVM 版本**：当前使用 LLVM 18，需要确保 API 兼容性
2. **平台限制**：目前仅支持 Ubuntu，macOS 支持需要额外配置
3. **资源限制**：GitHub Actions 免费账户有构建时间和并发限制
4. **缓存策略**：可以考虑添加依赖缓存以加速构建

## 未来改进

- [ ] 添加 RVV 后端测试（需要 RISC-V 工具链）
- [ ] 添加 macOS 平台支持
- [ ] 实现依赖缓存
- [ ] 添加代码覆盖率报告
- [ ] 添加性能基准测试
- [ ] 支持多版本 LLVM 测试矩阵

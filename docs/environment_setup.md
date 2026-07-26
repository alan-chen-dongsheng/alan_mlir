# Alan MLIR 环境配置指南

本文档详细说明如何配置 CPU 和 RVV 双后端的验证环境。

## 目录

- [必需工具](#必需工具)
- [CPU 后端环境](#cpu-后端环境)
- [RVV 后端环境](#rvv-后端环境)
- [环境变量配置](#环境变量配置)
- [验证安装](#验证安装)
- [常见问题](#常见问题)

---

## 必需工具

### 基础工具

| 工具 | 最低版本 | 用途 |
|------|---------|------|
| CMake | 3.20 | 构建系统 |
| LLVM/MLIR | 17.0 | MLIR 框架 |
| Clang | 随 LLVM | C/C++ 编译器 |
| Ninja | 1.10 | 构建工具(可选,推荐) |

### RVV 可选工具

| 工具 | 用途 | 必需性 |
|------|------|--------|
| riscv64-unknown-elf-gcc | RISC-V 交叉编译器 | RVV 测试必需 |
| Spike | RISC-V ISA 模拟器 | RVV 测试必需 |
| riscv-pk (pk) | Proxy Kernel | RVV 测试必需 |
| riscv64-unknown-elf-objdump | 反汇编工具 | RVV 调试推荐 |

---

## CPU 后端环境

CPU 后端仅需 LLVM/MLIR 工具链,配置相对简单。

### macOS 安装

```bash
# 1. 安装 LLVM (包含 MLIR 和 Clang)
brew install llvm

# 2. 配置环境变量
echo 'export PATH="/usr/local/opt/llvm/bin:$PATH"' >> ~/.zshrc
echo 'export LDFLAGS="-L/usr/local/opt/llvm/lib"' >> ~/.zshrc
echo 'export CPPFLAGS="-I/usr/local/opt/llvm/include"' >> ~/.zshrc
source ~/.zshrc

# 3. 验证安装
llvm-config --version
clang --version
```

### Linux (Ubuntu/Debian) 安装

```bash
# 1. 添加 LLVM 仓库
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt-add-repository "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-17 main"
sudo apt-get update

# 2. 安装 LLVM 和 MLIR
sudo apt-get install llvm-17 llvm-17-dev mlir-17-tools libmlir-17-dev

# 3. 设置符号链接(可选)
sudo ln -s /usr/bin/llvm-config-17 /usr/bin/llvm-config
sudo ln -s /usr/bin/clang-17 /usr/bin/clang

# 4. 验证安装
llvm-config --version
clang --version
```

### 构建项目

```bash
# 创建构建目录
mkdir -p build && cd build

# 配置 CMake (macOS)
cmake .. -DMLIR_DIR=$(brew --prefix llvm)/lib/cmake/mlir \
         -DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm

# 配置 CMake (Linux,假设默认安装路径)
cmake .. -DMLIR_DIR=/usr/lib/llvm-17/lib/cmake/mlir \
         -DLLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm

# 编译
cmake --build . -j$(nproc)

# 验证构建
./tools/alan-opt/alan-opt --help
```

### CPU 测试

```bash
# 运行 CPU 端到端测试
./tools/run_alan_cpu.sh test/Execution/Alan/eltwise_test.mlir

# 预期输出
# === Alan Eltwise CPU Execution ===
# ...
# ADD PASSED
# MUL PASSED
# MAX PASSED
# All tests PASSED!
```

---

## RVV 后端环境

RVV 后端需要 RISC-V 工具链和 Spike 模拟器,配置较为复杂。

### macOS 安装

```bash
# 1. 添加 RISC-V tap
brew tap riscv/riscv

# 2. 安装 RISC-V 工具链
brew install riscv-tools
# 这包括: riscv64-unknown-elf-gcc, riscv64-unknown-elf-binutils 等

# 3. 安装 Spike 模拟器
brew install riscv-isa-sim

# 4. 安装 Proxy Kernel
brew install riscv-pk

# 5. 查找 pk 的路径(后续需要配置)
which pk
# 通常是: /usr/local/Cellar/riscv-pk/main/riscv64-unknown-elf/bin/pk
# 或: /usr/local/bin/pk
```

### Linux (Ubuntu/Debian) 安装

```bash
# 1. 安装依赖
sudo apt-get install autoconf automake autotools-dev curl libmpc-dev \
    libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo \
    gperf libtool patchutils bc zlib1g-dev libexpat-dev

# 2. 克隆并构建 RISC-V GNU 工具链
git clone --recursive https://github.com/riscv/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv --with-arch=rv64gcv
make -j$(nproc)
cd ..

# 3. 添加到 PATH
echo 'export PATH="/opt/riscv/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 4. 构建 Spike 模拟器
git clone https://github.com/riscv/riscv-isa-sim.git
cd riscv-isa-sim
mkdir build && cd build
../configure --prefix=/opt/riscv --with-isa=rv64gcv
make -j$(nproc)
sudo make install
cd ../..

# 5. 构建 Proxy Kernel
git clone https://github.com/riscv/riscv-pk.git
cd riscv-pk
mkdir build && cd build
../configure --prefix=/opt/riscv --host=riscv64-unknown-elf \
    --with-arch=rv64gcv
make -j$(nproc)
sudo make install
cd ../..
```

### 验证 RVV 工具安装

```bash
# 检查 RISC-V GCC
riscv64-unknown-elf-gcc --version
# 应该显示: riscv64-unknown-elf-gcc (GCC) 12.x.x 或更高版本

# 检查 Spike
spike --help
# 应该显示: RISC-V ISA Simulator

# 检查 pk
ls -l /usr/local/Cellar/riscv-pk/main/riscv64-unknown-elf/bin/pk  # macOS
ls -l /opt/riscv/riscv64-unknown-elf/bin/pk  # Linux
```

### 配置环境变量

创建或编辑 `~/.zshrc` (macOS) 或 `~/.bashrc` (Linux):

```bash
# RISC-V 工具链路径
export RISCV_TOOLCHAIN="/usr/local"  # macOS Homebrew 默认
# export RISCV_TOOLCHAIN="/opt/riscv"  # Linux 自定义路径

# Spike 和 pk 路径
export SPIKE="/usr/local/bin/spike"
export PK="/usr/local/Cellar/riscv-pk/main/riscv64-unknown-elf/bin/pk"

# LLVM 工具链(如果使用非标准路径)
export LLVM_DIR="/usr/local/opt/llvm"
export PATH="$LLVM_DIR/bin:$PATH"

# Alan MLIR 项目路径
export ALAN_MLIR_ROOT="$HOME/workspace/alan_mlir/alan_mlir"
export PATH="$ALAN_MLIR_ROOT/build/tools:$PATH"
```

应用更改:

```bash
source ~/.zshrc  # 或 source ~/.bashrc
```

### RVV 测试

```bash
# 方式 1: 使用默认 pk 路径
./tools/run_alan_rvv_spike.sh test/Execution/Alan/eltwise_rvv_test.mlir

# 方式 2: 显式指定 pk 路径
PK=/usr/local/Cellar/riscv-pk/main/riscv64-unknown-elf/bin/pk \
  ./tools/run_alan_rvv_spike.sh test/Execution/Alan/eltwise_rvv_test.mlir

# 方式 3: 配置环境变量后
export PK=/path/to/pk
./tools/run_alan_rvv_spike.sh test/Execution/Alan/eltwise_rvv_test.mlir

# 预期输出
# === Alan Eltwise RVV Spike Execution ===
# ...
# ADD PASSED
# MUL PASSED
# MAX PASSED
# All RVV tests PASSED!
# SUCCESS: All RVV tests passed!
```

### 调试 RVV 问题

```bash
# 保留中间文件
KEEP_TEMP=1 ./tools/run_alan_rvv_spike.sh test/Execution/Alan/eltwise_rvv_test.mlir

# 查看生成的汇编
ls /tmp/alan_rvv_*/disasm.txt

# 检查 RVV 指令
grep -E '\b(vadd|vsub|vmul|vle|vse|vfadd|vfmul|vfmax|vsetvl)\.' \
  /tmp/alan_rvv_*/disasm.txt

# 手动运行 Spike 查看详细信息
spike --isa=rv64gcv --varch=vlen:256,elen:64 \
  /path/to/pk /tmp/alan_rvv_*/test_runner.elf
```

---

## 环境变量配置

### 完整环境变量示例

创建 `env.sh` 文件:

```bash
#!/bin/bash
# Alan MLIR 环境变量配置

# LLVM/MLIR 路径
export LLVM_DIR="/usr/local/opt/llvm"
export MLIR_DIR="$LLVM_DIR/lib/cmake/mlir"
export PATH="$LLVM_DIR/bin:$PATH"

# RISC-V 工具链路径
export RISCV_TOOLCHAIN="/usr/local"
export PATH="$RISCV_TOOLCHAIN/bin:$PATH"

# Spike 和 pk 路径
export SPIKE="/usr/local/bin/spike"
export PK="/usr/local/Cellar/riscv-pk/main/riscv64-unknown-elf/bin/pk"

# Spike 配置
export SPIKE_ISA="rv64gcv"
export SPIKE_VARCH="vlen:256,elen:64"

# 项目路径
export ALAN_MLIR_ROOT="$HOME/workspace/alan_mlir/alan_mlir"
export PATH="$ALAN_MLIR_ROOT/build/tools:$PATH"

echo "Alan MLIR 环境已配置"
echo "  LLVM: $LLVM_DIR"
echo "  RISC-V: $RISCV_TOOLCHAIN"
echo "  Spike: $SPIKE"
echo "  PK: $PK"
```

使用方法:

```bash
source env.sh
```

---

## 验证安装

### 快速验证脚本

创建 `verify_env.sh`:

```bash
#!/bin/bash
set -e

echo "=== 验证 Alan MLIR 环境 ==="
echo

# 检查 CMake
echo "1. 检查 CMake..."
cmake --version | head -1
echo "   ✓ CMake 已安装"
echo

# 检查 LLVM
echo "2. 检查 LLVM/MLIR..."
llvm-config --version
clang --version | head -1
echo "   ✓ LLVM 已安装"
echo

# 检查 alan-opt
echo "3. 检查 alan-opt..."
if [ -f "build/tools/alan-opt/alan-opt" ]; then
  ./build/tools/alan-opt/alan-opt --version 2>&1 | head -1 || echo "   ✓ alan-opt 已构建"
else
  echo "   ⚠ alan-opt 未构建,请先运行 cmake --build build"
fi
echo

# 检查 RISC-V 工具链(可选)
echo "4. 检查 RISC-V 工具链..."
if command -v riscv64-unknown-elf-gcc &> /dev/null; then
  riscv64-unknown-elf-gcc --version | head -1
  echo "   ✓ RISC-V GCC 已安装"
else
  echo "   ⚠ RISC-V GCC 未安装 (RVV 测试需要)"
fi

if command -v spike &> /dev/null; then
  spike --help 2>&1 | head -1 || echo "   ✓ Spike 已安装"
else
  echo "   ⚠ Spike 未安装 (RVV 测试需要)"
fi

if [ -n "$PK" ] && [ -f "$PK" ]; then
  echo "   ✓ PK 已配置: $PK"
else
  echo "   ⚠ PK 未配置 (RVV 测试需要)"
fi
echo

echo "=== 环境验证完成 ==="
```

运行验证:

```bash
chmod +x verify_env.sh
./verify_env.sh
```

---

## 常见问题

### 1. CMake 找不到 MLIR

**错误信息:**
```
CMake Error at CMakeLists.txt:XX (find_package):
  By not providing "FindMLIR.cmake" in CMAKE_MODULE_PATH this project has
  asked CMake to find a package configuration file provided by "MLIR"
```

**解决方案:**

```bash
# macOS
cmake .. -DMLIR_DIR=$(brew --prefix llvm)/lib/cmake/mlir \
         -DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm

# Linux
cmake .. -DMLIR_DIR=/usr/lib/llvm-17/lib/cmake/mlir \
         -DLLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm
```

### 2. 链接错误: 找不到 MLIR 库

**错误信息:**
```
ld: library not found for -lMLIRIR
```

**解决方案:**

```bash
# 确保 LDFLAGS 包含 LLVM 库路径
export LDFLAGS="-L$(brew --prefix llvm)/lib"

# 或在 CMake 中指定
cmake .. -DLLVM_BUILD_LIBRARY_DIR=$(brew --prefix llvm)/lib
```

### 3. Spike 找不到 pk

**错误信息:**
```
spike: error: unable to find proxy kernel
```

**解决方案:**

```bash
# 显式指定 pk 路径
PK=/path/to/pk ./tools/run_alan_rvv_spike.sh test/Execution/Alan/eltwise_rvv_test.mlir

# 或设置环境变量
export PK=/path/to/pk
```

### 4. RISC-V GCC 不支持 V 扩展

**错误信息:**
```
riscv64-unknown-elf-gcc: error: unknown arch 'rv64gcv'
```

**解决方案:**

确保安装的工具链支持 V 扩展:

```bash
# 检查支持的架构
riscv64-unknown-elf-gcc -print-supported-extensions

# 如果不支持,需要重新编译工具链
./configure --prefix=/opt/riscv --with-arch=rv64gcv
make -j$(nproc)
```

### 5. 构建时内存不足

**解决方案:**

```bash
# 减少并行编译数
cmake --build build -j2

# 或使用 Ninja 并限制内存
cmake .. -G Ninja
ninja -j2
```

### 6. alan-opt 运行时找不到动态库

**错误信息:**
```
dyld: Library not loaded: @rpath/libMLIRIR.dylib
```

**解决方案:**

```bash
# macOS
export DYLD_LIBRARY_PATH="$(brew --prefix llvm)/lib:$DYLD_LIBRARY_PATH"

# Linux
export LD_LIBRARY_PATH="/usr/lib/llvm-17/lib:$LD_LIBRARY_PATH"
```

---

## 完整安装清单

### macOS 完整安装

```bash
# 1. 基础工具
brew install cmake llvm ninja

# 2. RISC-V 工具链
brew tap riscv/riscv
brew install riscv-tools riscv-isa-sim riscv-pk

# 3. 配置环境变量
cat >> ~/.zshrc << 'EOF'
export PATH="/usr/local/opt/llvm/bin:$PATH"
export LDFLAGS="-L/usr/local/opt/llvm/lib"
export CPPFLAGS="-I/usr/local/opt/llvm/include"
export PK="/usr/local/Cellar/riscv-pk/main/riscv64-unknown-elf/bin/pk"
EOF
source ~/.zshrc

# 4. 构建项目
mkdir -p build && cd build
cmake .. -DMLIR_DIR=$(brew --prefix llvm)/lib/cmake/mlir \
         -DLLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm
cmake --build . -j8

# 5. 验证
cd ..
./tools/run_alan_cpu.sh test/Execution/Alan/eltwise_test.mlir
./tools/run_alan_rvv_spike.sh test/Execution/Alan/eltwise_rvv_test.mlir
```

### Linux 完整安装

```bash
# 1. 基础工具
sudo apt-get update
sudo apt-get install cmake ninja-build

# 2. LLVM 17
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo apt-add-repository "deb http://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-17 main"
sudo apt-get update
sudo apt-get install llvm-17 llvm-17-dev mlir-17-tools libmlir-17-dev

# 3. RISC-V 工具链(从源码编译)
sudo apt-get install autoconf automake autotools-dev curl libmpc-dev \
    libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo \
    gperf libtool patchutils bc zlib1g-dev libexpat-dev

git clone --recursive https://github.com/riscv/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv --with-arch=rv64gcv
sudo make -j$(nproc)
cd ..

# 4. Spike 和 pk
git clone https://github.com/riscv/riscv-isa-sim.git
cd riscv-isa-sim
mkdir build && cd build
../configure --prefix=/opt/riscv --with-isa=rv64gcv
sudo make -j$(nproc) install
cd ../..

git clone https://github.com/riscv/riscv-pk.git
cd riscv-pk
mkdir build && cd build
../configure --prefix=/opt/riscv --host=riscv64-unknown-elf --with-arch=rv64gcv
sudo make -j$(nproc) install
cd ../..

# 5. 配置环境变量
cat >> ~/.bashrc << 'EOF'
export PATH="/opt/riscv/bin:$PATH"
export PK="/opt/riscv/riscv64-unknown-elf/bin/pk"
EOF
source ~/.bashrc

# 6. 构建项目
mkdir -p build && cd build
cmake .. -DMLIR_DIR=/usr/lib/llvm-17/lib/cmake/mlir \
         -DLLVM_DIR=/usr/lib/llvm-17/lib/cmake/llvm
cmake --build . -j$(nproc)

# 7. 验证
cd ..
./tools/run_alan_cpu.sh test/Execution/Alan/eltwise_test.mlir
./tools/run_alan_rvv_spike.sh test/Execution/Alan/eltwise_rvv_test.mlir
```

---

## 参考资源

- [LLVM/MLIR 官方文档](https://mlir.llvm.org/)
- [RISC-V GNU 工具链](https://github.com/riscv/riscv-gnu-toolchain)
- [Spike 模拟器](https://github.com/riscv/riscv-isa-sim)
- [RISC-V Proxy Kernel](https://github.com/riscv/riscv-pk)

import re
import sys

mlir = sys.stdin.read()

# 1. Replace memref<NxT> in emitc.func signatures with !emitc.ptr<T>
def replace_memref_in_sig(match):
    func_name = match.group(1)
    sig = match.group(2)
    sig = re.sub(r'memref<\d+x([^>]+)>', r'!emitc.ptr<\1>', sig)
    return f'emitc.func @{func_name}({sig})'

mlir = re.sub(r'emitc\.func @(\w+)\(([^)]*)\)', replace_memref_in_sig, mlir)

# 2. Collect all unrealized_conversion_cast ops and their sources
casts = {}
cast_pattern = re.compile(
    r'^(\s+)(%\w+) = builtin\.unrealized_conversion_cast (%\w+) : ([^ ]+) to ([^ ]+)\s*$'
)

lines = mlir.split('\n')
new_lines = []
for line in lines:
    m = cast_pattern.match(line)
    if m:
        result = m.group(2)
        source = m.group(3)
        casts[result] = source
        continue
    new_lines.append(line)

mlir = '\n'.join(new_lines)

# Replace uses of cast results
for cast_val, src_val in sorted(casts.items(), key=lambda x: -len(x[0])):
    mlir = mlir.replace(cast_val, src_val)

# 3. Replace emitc.array types with emitc.ptr
mlir = re.sub(r'!emitc\.array<\d+x([^>]+)>', r'!emitc.ptr<\1>', mlir)

# 4. Fix subscript index type: replace 'index' with '!emitc.size_t' in subscript type sigs
#    Pattern: subscript ... : (!emitc.ptr<T>, index) ->
mlir = re.sub(
    r'(subscript\s+%[^\s]+\[[^\]]+\]\s*:\s*\([^,]+,\s*)index(\))',
    r'\1!emitc.size_t\2',
    mlir
)

print(mlir)

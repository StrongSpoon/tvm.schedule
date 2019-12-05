import tvm

n = 1024
k = 1024
m = 1024
A = tvm.placeholder((n, k), name='A')
B = tvm.placeholder((k, m), name='B')
K = tvm.reduce_axis((0, k), name='K')
C = tvm.compute((n, m), lambda i, j: tvm.sum(A[i, K] * B[K, j], axis=K), name='C')

s = tvm.create_schedule(C.op)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)

print(tvm.lower(s, [A, B, C], simple_mode=True))
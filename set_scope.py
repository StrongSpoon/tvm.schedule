import tvm

n = 1024
dtype = "float32"
A = tvm.placeholder((n, n), dtype=dtype, name='A')
k = tvm.reduce_axis((0, n), name='k')
B = tvm.compute((n,), lambda i: tvm.sum(A[i, k], axis=k), name='B')
C = tvm.compute((n,), lambda i: B[i] + 10, name='C')

s = tvm.create_schedule(C.op)

print(tvm.lower(s, [A, C], simple_mode=True))
print("---------cutting line---------")

s[B].set_scope('shared')

print(tvm.lower(s, [A, C], simple_mode=True))

import tvm

n = 1024
dtype = "float32"
k = tvm.reduce_axis((0, n), name='k')
A = tvm.placeholder((n, n), dtype=dtype, name='A')
B = tvm.compute((n,), lambda i: tvm.sum(A[i, k], axis=k), name='B')

s = tvm.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].prefetch(A, s[B].op.reduce_axis[0], 1)
print(tvm.lower(s, [A, B], simple_mode=True))
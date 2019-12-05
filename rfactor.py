import tvm

n = tvm.var('n')
k = tvm.reduce_axis((0, n), name='k')

A = tvm.placeholder((n, n), name='A')
B = tvm.compute((n,), lambda i: tvm.sum(A[i, k], axis=k), name='B')

s = tvm.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

BR = s.rfactor(B, B.op.reduce_axis[0])

print(tvm.lower(s, [A, B], simple_mode=True))
import tvm

n = 1024
A = tvm.placeholder((n,), name='A')
k = tvm.reduce_axis((0, n), name='k')

B = tvm.compute((1,), lambda i: tvm.sum(A[k], axis=k), name='B')

s = tvm.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)

print(tvm.lower(s, [A, B], simple_mode=True))
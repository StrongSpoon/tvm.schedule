import tvm
n = 1024
m = 1024

A = tvm.placeholder((n, m), name='A')
l = tvm.reduce_axis((0, m), name = 'l')

B = tvm.compute((n,), lambda i: tvm.sum(A[i, l], axis=l), name='B')

s = tvm.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].parallel(B.op.reduce_axis[0])
print(tvm.lower(s, [A, B], simple_mode=True))
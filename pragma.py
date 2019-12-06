import tvm

n = 1024
m = 1024
A = tvm.placeholder((n, m), name='A')
k = tvm.reduce_axis((0, n), name='k')
l = tvm.reduce_axis((0, m), name = 'l')

B = tvm.compute((n,), lambda i: tvm.sum(A[i, l], axis=l), name='B')

s = tvm.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor=4)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].pragma(ki, "unroll")

print(tvm.lower(s, [A, B], simple_mode=True))
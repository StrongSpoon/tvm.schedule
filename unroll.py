import tvm

n = 1024
A = tvm.placeholder((n, n), name='A')
B = tvm.placeholder((n, n), name='B')
C = tvm.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='C')

s = tvm.create_schedule(C.op)

xo, xi = s[C].split(s[C].op.axis[0], factor=4)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s[C].unroll(xi)

print(tvm.lower(s, [A, B, C], simple_mode=True))
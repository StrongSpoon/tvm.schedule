import tvm

n = tvm.var("n")
m = tvm.var("m")
A = tvm.placeholder((n, m), name='A')
B = tvm.placeholder((n, m), name='B')
C = tvm.compute((n, m), lambda i, j: A[i, j] + B[i, j], name='C')

s = tvm.create_schedule(C.op)

xo, xi = s[C].split(s[C].op.axis[0], factor=8)
yo, yi = s[C].split(s[C].op.axis[1], factor=16)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s[C].reorder(xo, yo, yi, xi)

print(tvm.lower(s, [A, B, C], simple_mode=True))
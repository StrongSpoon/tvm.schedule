import tvm

n = 1024
dtype = "float32"
A = tvm.placeholder((n, n), dtype=dtype, name='A')
B = tvm.placeholder((n, n), dtype=dtype, name='B')
C = tvm.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='C')
D = tvm.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='D')
E = C

s = tvm.create_schedule(C.op)
sch = tvm.create_schedule(D.op)
print(sch[D].same_as(s[C]))
print(s[C].same_as(s[E]))
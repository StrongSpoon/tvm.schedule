import tvm

n = 1024
k = tvm.te.reduce_axis((0, n), name='k')

A = tvm.te.placeholder((n, n), name='A')
B = tvm.te.placeholder((n, n), name='B')

D = tvm.te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='D')
E = tvm.te.compute((n, n), lambda i, j: D[i, j] + B[i, j], name='E')
F = tvm.te.compute((n,), lambda i: tvm.te.sum(E[i, k], axis=k), name='F')

s = tvm.te.create_schedule(F.op)

print(tvm.lower(s, [A, B, E], simple_mode=True))
print("---------cutting line---------")

g = s.create_group(outputs = E, inputs = [A, B], include_inputs=True)
g.compute_at(s[F], F.op.reduce_axis[0])

print(tvm.lower(s, [A, B, E], simple_mode=True))

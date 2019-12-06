import tvm

n = 1024
A = tvm.placeholder((n,), name='A')
k = tvm.reduce_axis((0, n), 'k')
B = tvm.compute((1,), lambda i: tvm.sum(A[k], axis=k), name='B')

s = tvm.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
BF = s.rfactor(B, ki)
tx = tvm.thread_axis("threadIdx.x")
s[BF].compute_at(s[B], s[B].op.reduce_axis[0])

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].set_store_predicate(tx.var.equal(0))

print(tvm.lower(s, [A, B], simple_mode=True))
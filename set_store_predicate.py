# TVM Reduction

import tvm
import numpy as np

from tvm.contrib import cc
from tvm.contrib import util

tgt = "cuda"
tgt_host = "llvm"

n = tvm.var("n")
A = tvm.placeholder((n), name='A')
k = tvm.reduce_axis((0, n), 'k')
B = tvm.compute((1,), lambda i: tvm.sum(A[k], axis=k), name='B')

s = tvm.create_schedule(B.op)

ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
BF = s.rfactor(B, ki)
tx = tvm.thread_axis("threadIdx.x")
s[B].bind(s[B].op.reduce_axis[0], tx)
s[BF].compute_at(s[B], s[B].op.reduce_axis[0])

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

s[B].set_store_predicate(tx.var.equal(0))

print(tvm.lower(s, [A, B], simple_mode=True))
exit(0)
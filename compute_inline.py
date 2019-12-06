import tvm

n = 1024
k = 3
pad = 2
A = tvm.placeholder((n, n), name='A')
W = tvm.placeholder((k, k), name='W')
m = (n - k + 2 * pad) + 1
Apad = tvm.compute((n + 2 * pad, n + 2 * pad),
                lambda yy, xx: tvm.if_then_else(
                    tvm.all(yy >= pad, yy < pad + n, xx >= pad, xx < pad + n), 
                    A[yy - pad, xx - pad], tvm.const(0., "float32")),
                    name='Apad')

ry = tvm.reduce_axis((0, k), name='ry')
rx = tvm.reduce_axis((0, k), name='rx')

B = tvm.compute((m, m),
                lambda yy, xx: 
                    tvm.sum(Apad[yy + ry, xx + rx] * W[ry, rx],
                    axis=[ry, rx]),
                    name='B')

s = tvm.create_schedule(B.op)

print(tvm.lower(s, [A, W, B], simple_mode=True))
print("---------cutting line---------")

s[Apad].compute_inline()

print(tvm.lower(s, [A, W, B], simple_mode=True))
exit(0)
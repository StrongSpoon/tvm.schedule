import tvm

b = tvm.var('batch')
n = tvm.var('n')
k = tvm.var('kernel')
in_channel = tvm.var('in_channel')
out_channel = tvm.var('out_channel')
pad = tvm.var('pad')
A = tvm.placeholder((n, n, in_channel, b), name='A')
W = tvm.placeholder((k, k, in_channel, out_channel), name='W')
m = (n - k + 2 * pad) + 1
Apad = tvm.compute((n + 2 * pad, n + 2 * pad, in_channel, b),
                lambda yy, xx, cc, nn: tvm.if_then_else(
                    tvm.all(yy >= pad, yy < pad + n, xx >= pad, xx < pad + n), 
                    A[yy - pad, xx - pad, cc, nn], tvm.const(0., "float32")),
                    name='Apad')
rc = tvm.reduce_axis((0, in_channel), name='rc')
ry = tvm.reduce_axis((0, k), name='ry')
rx = tvm.reduce_axis((0, k), name='rx')

B = tvm.compute((m, m, out_channel, b),
                lambda yy, xx, cc, nn: 
                    tvm.sum(Apad[yy + ry, xx + rx, rc, nn] * W[ry, rx, rc, cc],
                    axis=[ry, rx, rc]),
                    name='B')

s = tvm.create_schedule(B.op)

print(tvm.lower(s, [A, W, B], simple_mode=True))
print("---------cutting line---------")

s[Apad].compute_inline()

print(tvm.lower(s, [A, W, B], simple_mode=True))
exit(0)
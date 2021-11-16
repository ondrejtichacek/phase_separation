#!/bin/python

N = 50

a = 0.01
b = 0.99

d = (b - a) / (N - 1)

for i in range(N):
    print(a + d*i)
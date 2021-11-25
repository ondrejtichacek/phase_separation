#!/bin/python
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-N", type=int, default=50, help="")
parser.add_argument("--product", action="store_true", default=False, help="")

args = parser.parse_args()

N = args.N

a = 0.01
b = 0.99

d = (b - a) / (N - 1)

if args.product:
    for i in range(N):
        for j in range(N):
            print(f"{a + d*i} {a + d*j}") 
else:
    for i in range(N):
        print(a + d*i)
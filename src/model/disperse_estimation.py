import math


def square(a):
    b = a.copy()
    for i in range(len(b)):
        b[i] *= b[i]
    return b


def avg(A):
    return sum(A) / len(A)


def de(A):
    print('avg(',(square(A)), ')=  ',avg(square(A)), '-', avg(A)**2)
    return math.sqrt(avg(square(A)) - avg(A) ** 2) / avg(A)
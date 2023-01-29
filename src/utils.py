import math as m

MULTIPLES = ["B", "k{}B", "M{}B", "G{}B", "T{}B", "P{}B", "E{}B", "Z{}B", "Y{}B"]


def human_bytes(i, binary=False, precision=2):
    base = 1024 if binary else 1000
    multiple = m.trunc(m.log2(i) / m.log2(base))
    value = i / m.pow(base, multiple)
    suffix = MULTIPLES[multiple].format("i" if binary else "")
    return f"{value:.{precision}f} {suffix}"

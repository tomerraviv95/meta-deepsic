### code from site https://en.wikiversity.org/wiki/Reed%E2%80%93Solomon_codes_for_coders

import numpy as np

UINT8 = 256


def rs_generator_poly(nsym):
    '''Generate an irreducible generator polynomial (necessary to encode a message into Reed-Solomon)'''
    g = [1]
    for i in range(0, nsym):
        g = gf_poly_mul(g, [1, gf_pow(2, i)])
    return g


def gf_mult_noLUT(x, y, prim=0, field_charac_full=256, carryless=True):
    '''Galois Field integer multiplication using Russian Peasant Multiplication algorithm (faster than the standard multiplication + modular reduction).
    If prim is 0 and carryless=False, then the function produces the result for a standard integers multiplication (no carry-less arithmetics nor modular reduction).'''
    r = 0
    while y:  # while y is above 0
        if y & 1: r = r ^ x if carryless else r + x  # y is odd, then add the corresponding x to r (the sum of all x's corresponding to odd y's will give the final product). Note that since we're in GF(2), the addition is in fact an XOR (very important because in GF(2) the multiplication and additions are carry-less, thus it changes the result!).
        y = y >> 1  # equivalent to y // 2
        x = x << 1  # equivalent to x*2
        if prim > 0 and x & field_charac_full: x = x ^ prim  # GF modulo: if x >= 256 then apply modular reduction using the primitive polynomial (we just subtract, but since the primitive number can be above 256 then we directly XOR).

    return r


def gf_poly_add(p, q):
    r = [0] * max(len(p), len(q))
    for i in range(0, len(p)):
        r[i + len(r) - len(p)] = p[i]
    for i in range(0, len(q)):
        r[i + len(r) - len(q)] ^= q[i]
    return r


def gf_poly_mul(p, q):
    '''Multiply two polynomials, inside Galois Field'''
    # Pre-allocate the result array
    r = [0] * (len(p) + len(q) - 1)
    # Compute the polynomial multiplication (just like the outer product of two vectors,
    # we multiply each coefficients of p with all coefficients of q)
    for j in range(0, len(q)):
        for i in range(0, len(p)):
            r[i + j] ^= gf_mul(p[i], q[j])  # equivalent to: r[i + j] = gf_add(r[i+j], gf_mul(p[i], q[j]))
            # -- you can see it's your usual polynomial multiplication
    return r


def gf_mul(x, y):
    if x == 0 or y == 0:
        return 0
    return gf_exp[gf_log[x] + gf_log[y]]  # should be gf_exp[(gf_log[x]+gf_log[y])%255] if gf_exp wasn't oversized


def gf_inverse(x):
    return gf_exp[255 - gf_log[x]]  # gf_inverse(x) == gf_div(1, x)


def gf_sub(x, y):
    return x ^ y  # in binary galois field, subtraction is just the same as addition (since we mod 2)


def gf_pow(x, power):
    return gf_exp[(gf_log[x] * power) % 255]


def gf_poly_eval(poly, x):
    '''Evaluates a polynomial in GF(2^p) given the value for x. This is based on Horner's scheme for maximum efficiency.'''
    y = poly[0]
    for i in range(1, len(poly)):
        y = gf_mul(y, x) ^ poly[i]
    return y


def gf_div(x, y):
    if y == 0:
        raise ZeroDivisionError()
    if x == 0:
        return 0
    return gf_exp[(gf_log[x] + 255 - gf_log[y]) % 255]


def init_tables(prim=0x11d):
    global gf_exp, gf_log
    gf_exp = [0] * 2 * UINT8  # Create list of 512 elements. In Python 2.6+, consider using bytearray
    gf_log = [0] * UINT8
    '''Precompute the logarithm and anti-log tables for faster computation later, using the provided primitive polynomial.'''
    # prim is the primitive (binary) polynomial. Since it's a polynomial in the binary sense,
    # it's only in fact a single galois field value between 0 and 255, and not a list of gf values.

    gf_exp = [0] * 2 * UINT8  # anti-log (exponential) table
    gf_log = [0] * UINT8  # log table
    # For each possible value in the galois field 2^8, we will pre-compute the logarithm and anti-logarithm (exponential) of this value
    x = 1
    for i in range(0, UINT8 - 1):
        gf_exp[i] = x  # compute anti-log for this value and store it in a table
        gf_log[x] = i  # compute log at the same time
        x = gf_mult_noLUT(x, 2, prim)

        # If you use only generator==2 or a power of 2, you can use the following which is faster than gf_mult_noLUT():
        # x <<= 1 # multiply by 2 (change 1 by another number y to multiply by a power of 2^y)
        # if x & 0x100: # similar to x >= 256, but a lot faster (because 0x100 == 256)
        # x ^= prim # substract the primary polynomial to the current value (instead of 255, so that we get a unique set made of coprime numbers), this is the core of the tables generation

    # Optimization: double the size of the anti-log table so that we don't need to mod 255 to
    # stay inside the bounds (because we will mainly use this table for the multiplication of two GF numbers, no more).
    for i in range(UINT8 - 1, 2 * UINT8):
        gf_exp[i] = gf_exp[i - (UINT8 - 1)]


def gf_poly_scale(p, x):
    r = [0] * len(p)
    for i in range(0, len(p)):
        r[i] = gf_mul(p[i], x)
    return r


def convert_binary_to_field(array: np.ndarray):
    return np.packbits(array.reshape(-1, 8), axis=1).astype(np.uint8).reshape(-1)


def convert_field_to_binary(array: np.ndarray):
    return np.unpackbits(array.reshape(-1, 1), axis=1).astype(int).reshape(-1)

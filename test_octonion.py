
import pytest
from extended_numbers import O

def test_octonion_equality():
    a = O( (1.0, 0, 0, 0, 1.3, 0, 0, 0) )
    assert a == O( (1, 0, 0, 0, 1.3, 0, 0, 0) )

def test_octonion_nonequality():
    a = O( (1.4, 0, 1.0, 0, 2.0, -1.4, 0, 0) )
    assert a != O( (1.4, 0, 1.0, 0, 2.0, 1.4, 0, 0) )

def test_octonion_addition():
    a = O( (1, 0, 0, 0, 3, 2, 0) )
    b = O( (2.4, -1, 0, 0, 7, 0, 0) )
    c = O( (3.4, -1, 0, 0, 10, 2, 0) )
    assert a+b == c

def test_octonion_subtraction():
    a = O( (3.45, -6.459, 3.2, 0, 0, 0, 0, -1.3) )
    b = O( (0, 1.493, 6.3, 0, 0, 0, 0, 1.4) )
    c = O( (3.45, -7.952, -3.1, 0, 0, 0, 0, -2.7) )
    assert a-b == c

def test_octonion_const_mul():
    a = 2.5
    b = O( (0, 0, 1.2, -1, 0, 5.2, 0, 1.35) )
    c = O( (0, 0, 3, -2.5, 0, 13, 0, 3.375) )
    assert a*b == c

def test_octonion_const_rmul():
    a = 2.5
    b = O( (0, 0, 1.2, -1, 0, 5.2, 0, 1.35) )
    c = O( (0, 0, 3, -2.5, 0, 13, 0, 3.375) )
    assert b*a == c

def test_octonion_mul_scalar():
    a = O( (1., 0, 0, 0, 0, 0, 0, 0) )
    b = O( (4., 0, 0, 0, 0, 0, 0, 0) )
    c = O( (4., 0, 0, 0, 0, 0, 0, 0) )
    assert a.mul(b) == c

def test_octonion_mul_double_vector():
    a = O( (0, 0, 0, 0, 0, 3.2, 0, 0) )
    b = O( (0, 0, 0, 0, 0, -4, 0, 0) )
    c = O( (12.8, 0, 0, 0, 0, 0, 0, 0) )
    assert a.mul(b) == c

def test_octonion_mul_scalar_vector():
    a = O( (1.3, 0, 0, 0, 0, 0, 0, 0) )
    b = O( (0, 0, 0, 0, 0, 0, 4.23, 0) )
    c = O( (0, 0, 0, 0, 0, 0, 5.499, 0) )
    assert a.mul(b) == c

def test_octonion_noncommutative1():
    a = O( (0, 1.3, 0, 0, 0, 0, 0, 0) )
    b = O( (0, 0, 0, 4.23, 0, 0, 0, 0) )
    c = O( (0, 0, -5.499, 0, 0, 0, 0, 0) )
    assert a.mul(b) == c

def test_octonion_noncommutative2():
    a = O( (0, 1.3, 0, 0, 0, 0, 0, 0) )
    b = O( (0, 0, 0, 4.23, 0, 0, 0, 0) )
    c = O( (0, 0, 5.499, 0, 0, 0, 0, 0) )
    assert b.mul(a) == c

def test_octonion_mul_vector_comps():
    a = O( (0, 0, 0, 1.3, 0, 0, 0, 0) )
    b = O( (0, 0, 0, 0, 4.23, 0, 0, 0) )
    c = O( (0, 0, 0, 0, 0, 0, 0, 5.499) )
    assert a.mul(b) == c

def test_octonion_mul_general():
    a = O( (0.5, 0, 1., -3., 0, 0, 0.25, 2) )
    b = O( (3, -1, 0, 0, 0.5, 3, 0, 1.) )
    c =(O( (  1.5, -0.5,     0,   0, 0.25, 1.5,   0,  0.5) )+
        O( (    0,    0,     3,   1,    0,  -1, 0.5,    3) )+
        O( (    0,    0,     3,  -9,    3,   0,   9, -1.5) )+
        O( (    0,-0.25,-0.125,0.75,    0,   0,0.75,-0.25) )+
        O( (   -2,    0,    -6,  -1,    0,   0,   2,    6) ))
    assert a.mul(b) == c

def test_octonion_nonassociative1():
    a = O( (0, 2., 0,  0, 0,  0, 0,   0) )
    b = O( (0,  0, 0, 3., 0,  0, 0,   0) )
    c = O( (0,  0, 0,  0, 0, -2, 0,   0) )
    d = O( (0,  0, 0,  0, 0,  0, 0, 12) )
    assert a.mul(b).mul(c) == d

def test_octonion_nonassociative2():
    a = O( (0, 2., 0,  0, 0,  0, 0,   0) )
    b = O( (0,  0, 0, 3., 0,  0, 0,   0) )
    c = O( (0,  0, 0,  0, 0, -2, 0,   0) )
    d = O( (0,  0, 0,  0, 0,  0, 0, -12) )
    assert a.mul(b.mul(c)) == d



import pytest
from extended_numbers import Q

def test_quaternion_equality():
    a = Q( (1.0, 0, 1.3, 0) )
    assert a == Q( (1, 0, 1.3, 0) )

def test_quaternion_nonequality():
    a = Q( (1.4, 2.0, -1.4, 0) )
    assert a != Q( (1.4, 2.0, 1.4, 0) )

def test_quaternion_addition():
    a = Q( (1, 0, 3, 2) )
    b = Q( (2.4, -1, 7, 0) )
    c = Q( (3.4, -1, 10, 2) )
    assert a+b == c

def test_quaternion_subtraction():
    a = Q( (3.45, -6.459, 3.2, -1.3) )
    b = Q( (0, 1.493, 6.3, 1.4) )
    c = Q( (3.45, -7.952, -3.1, -2.7) )
    assert a-b == c

def test_quaternion_const_mul():
    a = 2.5
    b = Q( (1.2, -1, 5.2, 1.35) )
    c = Q( (3, -2.5, 13, 3.375) )
    assert a*b == c

def test_quaternion_const_rmul():
    a = 2.5
    b = Q( (1.2, -1, 5.2, 1.35) )
    c = Q( (3, -2.5, 13, 3.375) )
    assert b*a == c

def test_quaternion_mul_scalar():
    a = Q( (1., 0, 0, 0) )
    b = Q( (4., 0, 0, 0) )
    c = Q( (4., 0, 0, 0) )
    assert a*b == c

def test_quaternion_mul_double_vector():
    a = Q( (0, 3.2, 0, 0) )
    b = Q( (0, -4, 0, 0) )
    c = Q( (12.8, 0, 0, 0) )
    assert a*b == c

def test_quaternion_mul_scalar_vector():
    a = Q( (1.3, 0, 0, 0) )
    b = Q( (0, 0, 4.23, 0) )
    c = Q( (0, 0, 5.499, 0) )
    assert a*b == c

def test_quaternion_mul_vector():
    a = Q( (0, 1.3, 0, 0) )
    b = Q( (0, 0, 0, 4.23) )
    c = Q( (0, 0, -5.499, 0) )
    assert a*b == c

def test_quaternion_antisymmetric():
    a = Q( (0, 1.3, 0, 0) )
    b = Q( (0, 0, 0, 4.23) )
    c = Q( (0, 0, 5.499, 0) )
    assert b*a == c

def test_quaternion_mul_general():
    a = Q( (2.52, 0, 2, -48.529) )
    b = Q( (-59.259, 1.2384, 0, 42.1) )
    c = Q( (1893.73822, 87.320768, -178.6163136, 2979.395211) )
    assert a*b == c


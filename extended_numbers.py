'''class for quarternions '''

import numpy as np
import numbers 

class Q:
    
    def __init__(self, data):
        self.data = np.array(data)
    
    def __repr__(self):
        return "%s + %s i + %s j + %s k" %(self.data[0], self.data[1],\
                                           self.data[2], self.data[3])
    
    def __eq__(self, other):
        return np.allclose(self.data, other.data)
    
    def __add__(self, other):
        return Q( self.data + other.data )
    
    def __neg__(self):
        return Q( -self.data )
    
    def __sub__(self, other):
        return self + (-other)
    
    # conjugate of a quaternion
    def qstar(self, data):
        return Q((self.data[0], -self.data[1], -self.data[2], -self.data[3]))
    
    def __mul__(self, other):
        if isinstance(other, Q):
            return Q( (self.data[0]*other.data[0] - self.data[1]*other.data[1]  
                       -self.data[2]*other.data[2] -self.data[3]*other.data[3],
                       self.data[0]*other.data[1] + self.data[1]*other.data[0] 
                       -self.data[3]*other.data[2] + self.data[2]*other.data[3], 
                       self.data[2]*other.data[0] + self.data[3]*other.data[1] 
                       + self.data[0]*other.data[2] - self.data[1]*other.data[3], 
                       self.data[3]*other.data[0] - self.data[2]*other.data[1] \
                       +self.data[1]*other.data[2] + self.data[0]*other.data[3]) )
                
        elif isinstance(other, numbers.Number):
            return Q( other*self.data )
        else:
            raise ValueError("Can only multiply complex numbers by other complex numbers or by scalars")
            
    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return Q( other*self.data )
        else:
            raise ValueError("Can only multiply complex numbers by other complex numbers or by scalars")
            
            
'''Class for octonions'''
class O:
    
    
    def __init__(self, data):
        self.data = np.array(data)
    
    def __repr__(self):
        return "%s e0 + %s e1 + %s e2 + %s e3 + %s e4 + %s e5 +%s e6 + %s e7" \
            %(self.data[0],self.data[1],self.data[2],self.data[3], 
          self.data[4],self.data[5],self.data[6],self.data[7])
    
    def __eq__(self, other):
        return np.allclose(self.data, other.data)
    
    def __add__(self, other):
        return O( self.data + other.data )
    
    def __neg__(self):
        return O( -self.data )
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        if isinstance(other, numbers.Number):
            return O( other*self.data)
        else:
            raise ValueError("Can only multiply complex numbers by other complex numbers or by scalars")
            
    def __rmul__(self, other):
        if isinstance(other, numbers.Number):
            return O( other*self.data )
        else:
            raise ValueError("Can only multiply complex numbers by other complex numbers or by scalars")
            
    def mul(self, other):
        if isinstance(other, O):
            # Using Cayley-Dickinson 
            a, b = Q((self.data[:4])), Q((self.data[4:]))
            c, d = Q((other.data[:4])), Q((other.data[4:]))
            dstar = Q((d.data[0], -d.data[1], -d.data[2], -d.data[3]))
            cstar = Q((c.data[0], -c.data[1], -c.data[2], -c.data[3]))
            z1 = a*c - dstar*b
            z2 = d*a + b*cstar
            return O((z1.data[0], z1.data[1], z1.data[2], z1.data[3], z2.data[0],
                      z2.data[1], z2.data[2], z2.data[3]))
        else:
            raise ValueError("Can only multiply complex numbers by other complex numbers or by scalars")
            
            
            
            
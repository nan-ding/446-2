
import numpy as np
import scipy.fft

class Basis:

    def __init__(self, N, interval):
        self.N = N
        self.interval = interval


class Fourier(Basis):

    def __init__(self, N, interval=(0, 2*np.pi)):
        super().__init__(N, interval)

    def grid(self, scale=1):
        N_grid = int(np.ceil(self.N*scale))
        return np.linspace(self.interval[0], self.interval[1], num=N_grid, endpoint=False)

    def transform_to_grid(self, data, axis, dtype, scale=1):
        if dtype == np.complex128:
            return self._transform_to_grid_complex(data, axis, scale)
        elif dtype == np.float64:
            return self._transform_to_grid_real(data, axis, scale)
        else:
            raise NotImplementedError("Can only perform transforms for float64 or complex128")

    def transform_to_coeff(self, data, axis, dtype):
        if dtype == np.complex128:
            return self._transform_to_coeff_complex(data, axis)
        elif dtype == np.float64:
            return self._transform_to_coeff_real(data, axis)
        else:
            raise NotImplementedError("Can only perform transforms for float64 or complex128")

    def _transform_to_grid_complex(self, data, axis, scale):
        N = self.N 
        coeffs = np.zeros(scale*N, dtype=np.complex128)
        #fill the first N/2 and last N/2 terms by data
        coeffs[0:N//2] = data[0:N//2]
        coeffs[-1:-N//2-1:-1] = data[-1:-N//2-1:-1] 
        #normalization found by testing e^(ix)
        return scipy.fft.ifft(coeffs, axis=axis)*N*scale 


    def _transform_to_coeff_complex(self, data, axis):
        N=self.N
        coeffs = np.zeros(N, dtype=np.complex128)
        fft = scipy.fft.fft(data, axis=axis)/N #normalization
        coeffs[0: N//2] = fft[0:N//2]
        coeffs[-1:-1-N//2:-1] = fft[-1:-1-N//2:-1] #keeping the a_N/2 =0
       
        return coeffs

    def _transform_to_grid_real(self, data, axis, scale):
        N=self.N
        coeffs = np.zeros(scale*N//2+1, dtype=np.complex128)
        #fill the first N/2 terms with a_n and b_n
        coeffs.real[0:N//2] = data[0::2]
        coeffs.imag[0:N//2] = data[1::2]
        coeffs[0] = coeffs[0]*2 #normalization on a_0
        #normalization found by testing cosx and sinx
        return scipy.fft.irfft(coeffs, axis=axis)*scale*N/2 

    def _transform_to_coeff_real(self, data, axis):
        N=self.N
        #coefficients in complex form
        rfft = scipy.fft.rfft(data, axis=axis)/(N/2) #normalization by testing 
        coeffs = np.zeros(N, dtype=np.float128)
        coeffs[0::2] = rfft.real[0:N//2]
        coeffs[1::2] = rfft.imag[0:N//2]
        coeffs[1]=0 #b0=0
        coeffs[0] = coeffs[0]/2 #corresponding to the a_0 normalization in transform_to_grid
        return coeffs

class Domain:

    def __init__(self, bases):
        if isinstance(bases, Basis):
            # passed single basis
            self.bases = (bases, )
        else:
            self.bases = tuple(bases)
        self.dim = len(self.bases)

    @property
    def coeff_shape(self):
        return [basis.N for basis in self.bases]

    def remedy_scales(self, scales):
        if scales is None:
            scales = 1
        if not hasattr(scales, "__len__"):
            scales = [scales] * self.dim
        return scales


class Field:

    def __init__(self, domain, dtype=np.float64):
        self.domain = domain
        self.dtype = dtype
        self.data = np.zeros(domain.coeff_shape, dtype=dtype)
        self.coeff = np.array([True]*self.data.ndim)

    def towards_coeff_space(self):
        if self.coeff.all():
            # already in full coeff space
            return
        axis = np.where(self.coeff == False)[0][0]
        self.data = self.domain.bases[axis].transform_to_coeff(self.data, axis, self.dtype)
        self.coeff[axis] = True

    def require_coeff_space(self):
        if self.coeff.all():
            # already in full coeff space
            return
        else:
            self.towards_coeff_space()
            self.require_coeff_space()

    def towards_grid_space(self, scales=None):
        if not self.coeff.any():
            # already in full grid space
            return
        axis = np.where(self.coeff == True)[0][-1]
        scales = self.domain.remedy_scales(scales)
        self.data = self.domain.bases[axis].transform_to_grid(self.data, axis, self.dtype, scale=scales[axis])
        self.coeff[axis] = False

    def require_grid_space(self, scales=None):
        if not self.coeff.any(): 
            # already in full grid space
            return
        else:
            self.towards_grid_space(scales)
            self.require_grid_space(scales)




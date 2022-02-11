
import numpy as np
import spectral
from scipy import sparse

class KdVEquation:

    def __init__(self, domain, u):
         # store data we need for later, make M and L matrices
        self.dtype = u.dtype
        dtype = self.dtype
        self.u = u
        self.domain = domain
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.RHS = spectral.Field(domain, dtype=dtype) # 6u*dudx
        self.problem = spectral.InitialValueProblem(domain, [self.u], [self.RHS], dtype=dtype)

        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        if dtype == np.complex128:
            diag = -1j*x_basis.wavenumbers(dtype)**3
            p.L = sparse.diags(diag)
        elif dtype == np.float64:
            diag1 = np.zeros(x_basis.N-1)  #coeff for sin
            diag2 = np.zeros(x_basis.N -1) #coeff for cos
            diag1[::2] = x_basis.wavenumbers(dtype)[::2]**3
            diag2[::2] = -x_basis.wavenumbers(dtype)[::2]**3
            p.L = sparse.diags([diag1, diag2], offsets = (1, -1))
            
            

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        x_basis = self.domain.bases[0]
        u = self.u
        dudx = self.dudx
        RHS = self.RHS
        

        for i in range(num_steps):
            # need to calculate 6u*ux and put it into RHS
            u.require_coeff_space()
            dudx.require_coeff_space()
            if dudx.dtype == np.complex128:
                dudx.data = 1j*x_basis.wavenumbers(self.dtype)*u.data
            elif dudx.dtype == np.float64:
                diag1 = np.zeros(x_basis.N - 1) #readjusted coeffs for sin
                diag2 = np.zeros(x_basis.N - 1) #readjusted coeffs for cos
                diag1[::2] = -x_basis.wavenumbers(self.dtype)[::2]#-ksin(kx) after derivative
                diag2[::2] = x_basis.wavenumbers(self.dtype)[::2] 
                matrix = sparse.diags([diag1, diag2], offsets = (1, -1))
                dudx.data = matrix@ u.data
                
            u.require_grid_space(3/2)
            dudx.require_grid_space(3/2)
            RHS.require_grid_space(3/2)
            RHS.data = 6*u.data * dudx.data
            # u.require_coeff_space()
            # RHS.require_coeff_space()

            # take timestep
            ts.step(dt)

class SHEquation:

    def __init__(self, domain, u):
        # store data we need for later, make M and L matrices
        self.dtype = u.dtype
        dtype = self.dtype
        self.u = u
        self.domain = domain
        #self.dudx = spectral.Field(domain, dtype=dtype)
        self.RHS = spectral.Field(domain, dtype=dtype) # 6u*dudx
        self.problem = spectral.InitialValueProblem(domain, [self.u], [self.RHS], dtype=dtype)

        p = self.problem.pencils[0]

        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        diag = -x_basis.wavenumbers(dtype)**2 + x_basis.wavenumbers(dtype)**4
        p.L = sparse.diags(diag)

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        x_basis = self.domain.bases[0]
        u = self.u
        dudx = self.dudx
        RHS = self.RHS
        

        for i in range(num_steps):
            # need to calculate nonlinear RHS 
            u.require_coeff_space()
            u.require_grid_space(2)
            RHS.require_grid_space(2)
            RHS.data = -1.3*u.data + 1.8*u.data**2 - u.data**3
            u.require_coeff_space()
            RHS.require_coeff_space()

            # take timestep
            ts.step(dt)



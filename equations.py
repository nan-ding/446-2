
import spectral
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla



class SoundWaves:

    def __init__(self, domain, u, p, p0):
        dtype = u.dtype
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.p = p
        self.p_RHS = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u, p], [self.u_RHS, self.p_RHS],
                                                    num_BCs=2, dtype=dtype)
        self.p0 = p0
        
        ps = self.problem.pencils[0]
        self.N = N = domain.bases[0].N
        
        #derivative matrix
        diag = np.arange(N-1)+1
        self.D = D = sparse.diags(diag*2/3, offsets=1)

        diag0 = np.ones(N)/2
        diag0[0] = 1
        diag2 = -np.ones(N-2)/2
        self.C = C = sparse.diags((diag0, diag2), offsets=(0,2))
        
        M = sparse.csr_matrix((2*N+2,2*N+2))
        M[:N, :N] = C
        M[N:2*N, N:2*N] = C
        ps.M = M
        
        # L matrix
        BC_rows = np.zeros((2, 2*N))
        i = np.arange(N)
        BC_rows[0, :N] = (-1)**i
        BC_rows[1, :N] = (+1)**i

        cols = np.zeros((2*N,2))
        cols[  N-1, 0] = 1
        cols[2*N-1, 1] = 1
        corner = np.zeros((2,2))

        Z = np.zeros((N, N))
        L = sparse.bmat([[Z, D],
                         [D, Z]])
        L = sparse.bmat([[      L,   cols],
                         [BC_rows, corner]])
        L = L.tocsr()
        ps.L = L
        self.t = 0
        
        ps.M.eliminate_zeros()
        ps.L.eliminate_zeros()
        
        
        
    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        N = self.N
        u = self.u
        p = self.p
        #u_RHS = self.u_RHS
        dudx = self.dudx
        p_RHS = self.p_RHS
        p0 = self.p0
        
        
        for i in range(num_steps):
            # take a timestep
            
            #compute dudx
            u.require_coeff_space()
            p.require_coeff_space()
            p0.require_coeff_space
            #u_RHS.require_coeff_space()
            dudx.require_coeff_space()
            p_RHS.require_coeff_space()
            
            p_RHS.data = spla.spsolve(self.C, self.D @ u.data)
            # dudx.data[-1] = 0
            # dudx.data[-2] = 2*(N-1)*u.data[-1]
            # for n in range(N-3, 0, -1):
            #     dudx.data[n] = dudx.data[n+2] + 2*(n+1)*u.data[n+1]
            # dudx.data[0] = 0.5*dudx.data[2] + u.data[1]
            
            
            #dudx.data = spla.spsolve(self.C, self.D@u.data)
            dudx.require_grid_space()
            p_RHS.require_grid_space()
            
            p0.require_grid_space()
            p_RHS.data = (1 - p0.data)*p_RHS.data
           
            p_RHS.require_coeff_space()
            p_RHS.data = self.C @ p_RHS.data
            p_RHS.require_grid_space()
            
            ts.step(dt)
            self.t += dt

class CGLEquation:

    def __init__(self, domain, u):
        dtype = u.dtype
        self.u = u
        self.ux = ux = spectral.Field(domain, dtype=dtype)
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.ux_RHS = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u, ux], [self.u_RHS, self.ux_RHS],
                                                    num_BCs=2, dtype=dtype)
        
        self.x_basis = domain.bases[0]
        # interval 0 to L normalization in D
        L = self.x_basis.interval[-1]
        
        
        b = 0.5
        
        p = self.problem.pencils[0]
        self.N = N = domain.bases[0].N
        
        #derivative matrix
        diag = np.arange(N-1)+1
        self.D = D = sparse.diags(diag*2/L, offsets=1)

        diag0 = np.ones(N)/2
        diag0[0] = 1
        diag2 = -np.ones(N-2)/2
        self.C = C = sparse.diags((diag0, diag2), offsets=(0,2))
        
        M = sparse.csr_matrix((2*N+2,2*N+2))
        M[N:2*N, 0:N] = C
        p.M = M
        
        # L matrix
        BC_rows = np.zeros((2, 2*N))
        i = np.arange(N)
        BC_rows[0, :N] = (-1)**i
        BC_rows[1, :N] = (+1)**i

        cols = np.zeros((2*N,2))
        cols[  N-1, 0] = 1
        cols[2*N-1, 1] = 1
        corner = np.zeros((2,2))

        #Z = np.zeros((N, N)) 
        # diagz = -np.ones(N)
        # Z = sparse.diags(diagz) #takes care of the -u part
        L = sparse.bmat([[D, -C],
                         [-C, -(1+b*1j)*D]]) #need to normalize the matrix
        L = sparse.bmat([[      L,   cols],
                         [BC_rows, corner]])
        #L = L.tocsr()
        p.L = L
        self.t = 0
        
        p.M.eliminate_zeros()
        p.L.eliminate_zeros()
        
        

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        N = self.N
        u = self.u
        u_RHS = self.u_RHS
        ux_RHS = self.ux_RHS
        ux = self.ux
        c = -1.76
        BC_func = lambda t: [0*t, 0*t]

        for i in range(num_steps):
            # take a timestep
            
            #compute dudx
            u.require_coeff_space()
            #ux.require_grid_space()
            u.require_grid_space(scales=3)
            
            u_RHS.require_coeff_space()
            ux_RHS.require_coeff_space()
            ux_RHS.require_grid_space(scales=3)
            
        
            ux_RHS.data = np.absolute(u.data)**2*u.data  #the square of u in grid space
            
            
            ux_RHS.require_coeff_space()
            ux_RHS.data = -(1+c*1j)*self.C @ ux_RHS.data
            u.require_coeff_space()
            ux_RHS.require_grid_space(scales=3/2)
        
            
            ts.step(dt, BC_func(self.t))
            self.t += dt
            
           
            

class BurgersEquation: 

    def __init__(self, domain, u, nu):
        dtype = u.dtype
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u], [self.u_RHS], dtype=dtype)
        
        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        D = x_basis.derivative_matrix(dtype)
        p.L = -nu*D@D
        
    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        dudx = self.dudx
        u_RHS = self.u_RHS
        for i in range(num_steps):
            dudx.require_coeff_space()
            u.require_coeff_space()
            dudx.data = u.differentiate(0)
            u.require_grid_space()
            dudx.require_grid_space()
            u_RHS.require_grid_space()
            u_RHS.data = -u.data*dudx.data
            ts.step(dt)


class KdVEquation:
    
    def __init__(self, domain, u):
        dtype = u.dtype
        self.dealias = 3/2
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.dudx = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u], [self.u_RHS], dtype=dtype)
        
        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        D = x_basis.derivative_matrix(dtype)
        p.L = D@D@D
        
    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        dudx = self.dudx
        u_RHS = self.u_RHS
        for i in range(num_steps):
            dudx.require_coeff_space()
            u.require_coeff_space()
            dudx.data = u.differentiate(0)
            u.require_grid_space(scales=self.dealias)
            dudx.require_grid_space(scales=self.dealias)
            u_RHS.require_grid_space(scales=self.dealias)
            u_RHS.data = 6*u.data*dudx.data
            ts.step(dt)


class SHEquation:

    def __init__(self, domain, u):
        dtype = u.dtype
        self.dealias = 2
        self.u = u
        self.u_RHS = spectral.Field(domain, dtype=dtype)
        self.problem = spectral.InitialValueProblem(domain, [u], [self.u_RHS], dtype=dtype)

        p = self.problem.pencils[0]
        x_basis = domain.bases[0]
        I = sparse.eye(x_basis.N)
        p.M = I
        D = x_basis.derivative_matrix(dtype)
        op = I + D@D
        p.L = op @ op + 0.3*I

    def evolve(self, timestepper, dt, num_steps):
        ts = timestepper(self.problem)
        u = self.u
        u_RHS = self.u_RHS
        for i in range(num_steps):
            u.require_coeff_space()
            u.require_grid_space(scales=self.dealias)
            u_RHS.require_grid_space(scales=self.dealias)
            u_RHS.data = 1.8*u.data**2 - u.data**3
            ts.step(dt)




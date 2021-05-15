
import numpy as np

class PDMA:
    """
    Pentadiagonal matrix solver from shenfun library

    Parameters
    ----------
        mat : SparseMatrix
            Pentadiagonal matrix with diagonals in offsets
            -4, -2, 0, 2, 4
        neumann : bool, optional
            Whether matrix represents a Neumann problem, where
            the first index is known as the mean value and we
            solve for slice(1, N-3).
            If `mat` is a :class:`.SpectralMatrix`, then the
            `neumann` keyword is ignored and the information
            extracted from the matrix.
    """

    def __init__(self, mat, neumann=False):
        self.mat = mat
        self.N = 0
        self.d0 = np.zeros(0)
        self.d1 = None
        self.d2 = None
        self.A = None
        self.L = None
        self.neumann = neumann

    def init(self):
        """Initialize and allocate solver"""
        B = self.mat
        shape = self.mat.shape[1]
        # Broadcast in case diagonal is simply a constant.
        self.d0 = np.broadcast_to(np.diag(B,0), shape).copy()#*B.scale
        self.d1 = np.broadcast_to(np.diag(B,2), shape-2).copy()#*B.scale
        self.d2 = np.broadcast_to(np.diag(B,4), shape-4).copy()#*B.scale
        if self.neumann:
            self.d0[0] = 1
            self.d1[0] = 0
            self.d2[0] = 0
        self.l1 = np.broadcast_to(np.diag(B,-2), shape-2).copy()#*B.scale
        self.l2 = np.broadcast_to(np.diag(B,-4), shape-4).copy()#*B.scale
        self.PDMA_LU(self.l2, self.l1, self.d0, self.d1, self.d2)

    @staticmethod
    def PDMA_LU(a, b, d, e, f): # pragma: no cover
        """LU decomposition"""
        n = d.shape[0]
        m = e.shape[0]
        k = n - m

        for i in range(n-2*k):
            lam = b[i]/d[i]
            d[i+k] -= lam*e[i]
            e[i+k] -= lam*f[i]
            b[i] = lam
            lam = a[i]/d[i]
            b[i+k] -= lam*e[i]
            d[i+2*k] -= lam*f[i]
            a[i] = lam

        i = n-4
        lam = b[i]/d[i]
        d[i+k] -= lam*e[i]
        b[i] = lam
        i = n-3
        lam = b[i]/d[i]
        d[i+k] -= lam*e[i]
        b[i] = lam


    @staticmethod
    def PDMA_Solve(a, b, d, e, f, h, axis=0): # pragma: no cover
        """Symmetric solve (for testing only)"""
        n = d.shape[0]
        bc = h

        bc[2] -= b[0]*bc[0]
        bc[3] -= b[1]*bc[1]
        for k in range(4, n):
            bc[k] -= (b[k-2]*bc[k-2] + a[k-4]*bc[k-4])

        bc[n-1] /= d[n-1]
        bc[n-2] /= d[n-2]
        bc[n-3] = (bc[n-3]-e[n-3]*bc[n-1])/d[n-3]
        bc[n-4] = (bc[n-4]-e[n-4]*bc[n-2])/d[n-4]
        for k in range(n-5, -1, -1):
            bc[k] = (bc[k]-e[k]*bc[k+2]-f[k]*bc[k+4])/d[k]

    def solve(self,x):
        self.PDMA_Solve(self.l2, self.l1, self.d0, self.d1, self.d2, x, axis=0)

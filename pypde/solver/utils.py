import scipy.sparse as sp

def tosparse(A,type="csr"):
    assert type in ["csr","csc"], \
    "to sparse implements type='csc/csr' "
    if type=="csr":
        return sp.csr_matrix(A)
    if type=="csc":
        return sp.csr_matrix(A)

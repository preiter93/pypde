import numpy as np
from .utils import tosparse


def pseudo_inv(u, D=0, **kwargs):
    import pypde.bases.spectralbase as sb

    """
    Returns Pseudoinverse of spectral derivative matrix

    This is used to make systems banded and efficient.

    NOTE:
        Only supports Chebyshev Bases "CH"

    Input
        u:  MetaBase
        D:  int
            Order of derivative

    Output
        matrix
    """
    assert isinstance(u, sb.MetaBase)
    return PseudoInvKnown().check(u, D)


class PseudoInvKnown:
    """
    This class contains familiar inversese of inner products

    Use:
    InnerInvKnown().check(u,v,ku,kv)

    """

    inverted = False

    @property
    def dict(self):
        return {
            # Chebyshev
            "CH^0": self.chebyshev_d0_inv,
            "CH^1": self.chebyshev_d1_inv,
            "CH^2": self.chebyshev_d2_inv,
        }

    def check(self, u, ku):
        """
        Input:
            u:  MetaBase
            ku: Integers (Order of derivative)
        """
        assert all(hasattr(i, "id") for i in [u])
        assert u.id == "CH", "InnerInv supports only Chebyshev at the moment"

        key = self.generate_key(u.id, ku)

        # Lookup Key
        if key in self.dict:
            value = self.dict[key](u=u, ku=ku)
            return value
        else:
            raise ValueError("Key not found in inner_inv.")

    def generate_key(self, idu, ku):
        return "{:2s}^{:1d}".format(idu, ku)

    # ----- Collection of known inverses ------

    def chebyshev_d0_inv(self, u=None, **kwargs):
        """
        Returns identitiy matrix
        """
        return tosparse(np.eye(u.N)).toarray()

    # @staticmethod
    def chebyshev_d1_inv(self, u=None, k=1, **kwargs):
        """
        Pseudoinverse
        D1
        First row can be discarded
        """
        from .dmsuite import pseudoinverse_spectral as pis

        return tosparse(pis(u.N, k)).toarray()

    def chebyshev_d2_inv(self, u=None, **kwargs):
        """
        Pseudoinverse:
        D2
        First two rows can be discarded
        """
        return self.chebyshev_d1_inv(u, 2)

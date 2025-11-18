
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from contimod_graphene import bernal
from contimod_graphene.params import graphene_params_BLG

# Enable x64 for precision checks
jax.config.update("jax_enable_x64", True)

class TestBernal(unittest.TestCase):
    def setUp(self):
        self.params = dict(graphene_params_BLG)
        self.params["U"] = 0.0

    def test_hamiltonian_shape_and_hermiticity(self):
        N_layers = 4
        h_func = bernal.get_hamiltonian(N_layers=N_layers, params=self.params)
        
        # Test at a generic k-point
        kx, ky = 0.1, -0.1
        h_mat = h_func(kx, ky)
        
        self.assertEqual(h_mat.shape, (2*N_layers, 2*N_layers))
        
        # Check hermiticity
        self.assertTrue(jnp.allclose(h_mat, h_mat.T.conj()), "Hamiltonian is not Hermitian")

    def test_hamiltonian_ll_shape_and_hermiticity(self):
        N_layers = 4
        Ncut = 15
        h_ll_func = bernal.get_hamiltonian_LL(N_layers=N_layers, Ncut=Ncut, params=self.params)
        
        B = 5.0
        h_ll_mat = h_ll_func(B)
        
        # Expected size: N_layers * (2*Ncut - 1)
        expected_size = N_layers * (2*Ncut - 1)
        self.assertEqual(h_ll_mat.shape, (expected_size, expected_size))
        
        # Check hermiticity
        self.assertTrue(jnp.allclose(h_ll_mat, h_ll_mat.T.conj()), "LL Hamiltonian is not Hermitian")

if __name__ == '__main__':
    unittest.main()


import unittest
import jax
import jax.numpy as jnp
import numpy as np
from contimod_graphene import rhombohedral
from contimod_graphene.params import graphene_params_BLG

# Enable x64 for precision checks
jax.config.update("jax_enable_x64", True)

class TestRhombohedral(unittest.TestCase):
    def setUp(self):
        self.params = dict(graphene_params_BLG)
        self.params["U"] = 0.0

    def test_hamiltonian_shape_and_hermiticity(self):
        n_layers = 3
        h_func = rhombohedral.get_hamiltonian(n_layers=n_layers, params=self.params)
        
        # Test at a generic k-point
        kx, ky = 0.1, 0.2
        h_mat = h_func(kx, ky)
        
        self.assertEqual(h_mat.shape, (2*n_layers, 2*n_layers))
        
        # Check hermiticity
        self.assertTrue(jnp.allclose(h_mat, h_mat.T.conj()), "Hamiltonian is not Hermitian")

    def test_hamiltonian_ll_shape_and_hermiticity(self):
        n_layers = 3
        n_cut = 20
        h_ll_func = rhombohedral.get_hamiltonian_LL(n_layers=n_layers, n_cut=n_cut, params=self.params)
        
        B = 10.0
        h_ll_mat = h_ll_func(B)
        
        # Expected size: n_layers * (2*n_cut - 1)
        expected_size = n_layers * (2*n_cut - 1)
        self.assertEqual(h_ll_mat.shape, (expected_size, expected_size))
        
        # Check hermiticity
        self.assertTrue(jnp.allclose(h_ll_mat, h_ll_mat.T.conj()), "LL Hamiltonian is not Hermitian")

    def test_2band_hamiltonian(self):
        n_layers = 3
        h_2band_func = rhombohedral.get_2band_hamiltonian(n_layers=n_layers, params=self.params)
        
        kx, ky = 0.05, 0.05
        h_2band = h_2band_func(kx, ky)
        
        self.assertEqual(h_2band.shape, (2, 2))
        self.assertTrue(jnp.allclose(h_2band, h_2band.T.conj()), "2-band Hamiltonian is not Hermitian")

if __name__ == '__main__':
    unittest.main()

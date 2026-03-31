
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
        n_layers = 4
        h_func = bernal.get_hamiltonian(n_layers=n_layers, params=self.params)
        
        # Test at a generic k-point
        kx, ky = 0.1, -0.1
        h_mat = h_func(kx, ky)
        
        self.assertEqual(h_mat.shape, (2*n_layers, 2*n_layers))
        
        # Check hermiticity
        self.assertTrue(jnp.allclose(h_mat, h_mat.T.conj()), "Hamiltonian is not Hermitian")

    def test_hamiltonian_ll_shape_and_hermiticity(self):
        n_layers = 4
        n_cut = 15
        h_ll_func = bernal.get_hamiltonian_LL(n_layers=n_layers, n_cut=n_cut, params=self.params)
        
        B = 5.0
        h_ll_mat = h_ll_func(B)
        
        # Expected size: n_layers * (2*n_cut - 1)
        expected_size = n_layers * (2*n_cut - 1)
        self.assertEqual(h_ll_mat.shape, (expected_size, expected_size))
        
        # Check hermiticity
        self.assertTrue(jnp.allclose(h_ll_mat, h_ll_mat.T.conj()), "LL Hamiltonian is not Hermitian")

    def test_zero_field_delta_enters_as_global_sublattice_asymmetry(self):
        params = dict(self.params)
        params.update({"gamma2": 0.0, "gamma3": 0.0, "gamma4": 0.0, "gamma5": 0.0, "delta": 0.0, "Delta": 18.0})

        h_mat = bernal.hamiltonian(0.0, 0.0, n_layers=3, params=params)
        np.testing.assert_allclose(
            np.diag(np.asarray(h_mat)),
            np.array([9.0, -9.0, 9.0, -9.0, 9.0, -9.0]),
            atol=1e-10,
        )

    def test_zero_field_next_nearest_layer_blocks_swap_gamma2_gamma5_on_even_links(self):
        params = dict(self.params)
        params.update(
            {
                "gamma0": 0.0,
                "gamma1": 0.0,
                "gamma2": 12.0,
                "gamma3": 0.0,
                "gamma4": 0.0,
                "gamma5": 34.0,
                "U": 0.0,
                "Delta": 0.0,
                "delta": 0.0,
            }
        )

        h_mat = np.asarray(bernal.hamiltonian(0.0, 0.0, n_layers=4, params=params))

        np.testing.assert_allclose(h_mat[0:2, 4:6], np.diag([6.0, 17.0]), atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(h_mat[2:4, 6:8], np.diag([17.0, 6.0]), atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(h_mat[4:6, 0:2], np.diag([6.0, 17.0]), atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(h_mat[6:8, 2:4], np.diag([17.0, 6.0]), atol=1e-10, rtol=1e-10)

if __name__ == '__main__':
    unittest.main()

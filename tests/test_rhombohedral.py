
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from contimod_graphene import rhombohedral
from contimod_graphene.params import graphene_params_TLG

# Enable x64 for precision checks
jax.config.update("jax_enable_x64", True)

class TestRhombohedral(unittest.TestCase):
    def setUp(self):
        self.params = dict(graphene_params_TLG)
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

    def test_default_zero_field_surface_uses_tlg_preset(self):
        kx, ky = 0.07, -0.03
        h_default = rhombohedral.hamiltonian(kx, ky)
        h_explicit = rhombohedral.hamiltonian(kx, ky, n_layers=3, params=graphene_params_TLG)
        self.assertTrue(jnp.allclose(h_default, h_explicit))

    def test_default_getters_target_trilayer_abc(self):
        kx, ky = 0.02, 0.01

        h_default = rhombohedral.get_hamiltonian()(kx, ky)
        h_explicit = rhombohedral.get_hamiltonian(n_layers=3, params=graphene_params_TLG)(kx, ky)
        self.assertEqual(h_default.shape, (6, 6))
        self.assertTrue(jnp.allclose(h_default, h_explicit))

        h2_default = rhombohedral.get_2band_hamiltonian()(kx, ky)
        h2_explicit = rhombohedral.get_2band_hamiltonian(n_layers=3, params=graphene_params_TLG)(kx, ky)
        self.assertEqual(h2_default.shape, (2, 2))
        self.assertTrue(jnp.allclose(h2_default, h2_explicit))

    def test_default_ll_surface_uses_tlg_preset(self):
        h_ll_default = rhombohedral.get_hamiltonian_LL(n_cut=8)(10.0)
        h_ll_explicit = rhombohedral.get_hamiltonian_LL(
            n_layers=3,
            n_cut=8,
            params=graphene_params_TLG,
        )(10.0)
        self.assertEqual(h_ll_default.shape, (45, 45))
        self.assertTrue(np.allclose(h_ll_default, h_ll_explicit))

    def test_delta_enters_zero_field_as_inversion_even_layer_profile(self):
        params = dict(self.params)
        params.update({"gamma2": 0.0, "gamma3": 0.0, "gamma4": 0.0, "U": 0.0, "Delta": 6.0, "delta": 0.0})

        h_tlg = np.asarray(rhombohedral.hamiltonian(0.0, 0.0, n_layers=3, params=params))
        np.testing.assert_allclose(np.diag(h_tlg), np.array([6.0, 6.0, -12.0, -12.0, 6.0, 6.0]), atol=1e-10)

        h_4lg = np.asarray(rhombohedral.hamiltonian(0.0, 0.0, n_layers=4, params=params))
        np.testing.assert_allclose(
            np.diag(h_4lg),
            np.array([6.0, 6.0, -6.0, -6.0, -6.0, -6.0, 6.0, 6.0]),
            atol=1e-10,
        )

    def test_delta_enters_ll_surface_when_hoppings_vanish(self):
        params = dict(self.params)
        params.update(
            {
                "gamma0": 0.0,
                "gamma1": 0.0,
                "gamma2": 0.0,
                "gamma3": 0.0,
                "gamma4": 0.0,
                "U": 0.0,
                "Delta": 5.0,
                "delta": 0.0,
            }
        )

        h_ll = rhombohedral.hamiltonian_LL(1.0, n_layers=3, n_cut=6, params=params)
        evals = np.linalg.eigvalsh(h_ll)
        np.testing.assert_allclose(
            evals,
            np.sort(np.concatenate([np.full(11, -10.0), np.full(22, 5.0)])),
            atol=1e-10,
        )

    def test_delta_is_currently_ignored_by_all_rhombohedral_surfaces(self):
        base = dict(self.params)
        shifted = dict(self.params)
        shifted["delta"] = self.params["delta"] + 37.5

        np.testing.assert_allclose(
            np.asarray(rhombohedral.hamiltonian(0.02, -0.01, n_layers=3, params=base)),
            np.asarray(rhombohedral.hamiltonian(0.02, -0.01, n_layers=3, params=shifted)),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(rhombohedral.hamiltonian_2bands(0.02, -0.01, n_layers=3, params=base)),
            np.asarray(rhombohedral.hamiltonian_2bands(0.02, -0.01, n_layers=3, params=shifted)),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            rhombohedral.hamiltonian_LL(4.0, n_layers=3, n_cut=8, params=base),
            rhombohedral.hamiltonian_LL(4.0, n_layers=3, n_cut=8, params=shifted),
            atol=1e-10,
        )

if __name__ == '__main__':
    unittest.main()

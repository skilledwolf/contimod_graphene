import pathlib

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import contimod_graphene as cm_graphene


DOCS_ROOT = pathlib.Path(__file__).resolve().parent
STATIC_DIR = DOCS_ROOT / "_static"


def _ensure_static_dir() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)


def generate_abc_trilayer_bandstructure() -> None:
    params = dict(cm_graphene.params.graphene_params_TLG)
    params["U"] = 10.0

    h_full = cm_graphene.rhombohedral.get_hamiltonian(n_layers=3, params=params)
    h_low = cm_graphene.rhombohedral.get_2band_hamiltonian(n_layers=3, params=params)

    k_lin = 0.28 * jnp.linspace(-0.5, 0.5, 400)
    ks = jnp.stack([k_lin, jnp.zeros_like(k_lin)], axis=-1)

    hs_full = jax.vmap(h_full, in_axes=(0, 0))(*ks.T)
    bands_full = jnp.linalg.eigvalsh(hs_full)

    hs_low = jax.vmap(h_low, in_axes=(0, 0))(*ks.T)
    bands_low = jnp.linalg.eigvalsh(hs_low)

    fig, ax = plt.subplots()
    for band in bands_full.T:
        ax.plot(k_lin, band, color="black", linewidth=1.5, alpha=0.5)
    for band in bands_low.T:
        ax.plot(k_lin, band, color="red", linewidth=1.5, linestyle="--", alpha=0.9)

    ax.set_xlabel(r"$k_x\,a$")
    ax.set_ylabel("Energy [meV]")
    ax.set_ylim(-80, 80)

    output = STATIC_DIR / "abc_trilayer_bandstructure.png"
    fig.savefig(output, bbox_inches="tight", dpi=150)
    plt.close(fig)


def generate_blg_landau_level_fan() -> None:
    params = dict(cm_graphene.params.graphene_params_BLG)
    params["U"] = 0.0

    h_ll = cm_graphene.rhombohedral.get_hamiltonian_LL(
        n_layers=2,
        n_cut=40,
        params=params,
    )

    b_values = jnp.linspace(0.5, 9.0, 80)

    eigvals = []
    for b in b_values:
        h = h_ll(b)
        e, _ = jnp.linalg.eigh(h)
        eigvals.append(e)

    fig, ax = plt.subplots()
    for b, e in zip(b_values, eigvals):
        ax.plot(jnp.full_like(e, b), e, "k.", ms=2)

    ax.set_xlabel(r"Magnetic field $B$ [T]")
    ax.set_ylabel("Energy [meV]")
    ax.set_ylim(-15, 15)

    output = STATIC_DIR / "blg_landau_level_fan.png"
    fig.savefig(output, bbox_inches="tight", dpi=150)
    plt.close(fig)


def generate_aba_trilayer_bandstructure() -> None:
    graphene_params_aba = {
        "gamma0": 3100,
        "gamma1": 370,
        "gamma2": -19,
        "gamma3": 315,
        "gamma4": 140,
        "gamma5": 20,
        "U": 0.0,
        "Delta": 18.5,
        "delta": 3.8,
    }

    params_aba = dict(graphene_params_aba)
    h_aba = cm_graphene.bernal.get_hamiltonian(n_layers=3, params=params_aba)

    k_lin = 0.28 * jnp.linspace(-0.5, 0.5, 400)
    ks = jnp.stack([k_lin, jnp.zeros_like(k_lin)], axis=-1)

    hs = jax.vmap(h_aba, in_axes=(0, 0))(*ks.T)
    bands = jnp.linalg.eigvalsh(hs)

    fig, ax = plt.subplots()
    for band in bands.T:
        ax.plot(k_lin, band, color="black", linewidth=1.5, alpha=0.5)

    ax.set_xlabel(r"$k_x\,a$")
    ax.set_ylabel("Energy [meV]")
    ax.set_ylim(-80, 80)

    output = STATIC_DIR / "aba_trilayer_bandstructure.png"
    fig.savefig(output, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    _ensure_static_dir()
    generate_abc_trilayer_bandstructure()
    generate_blg_landau_level_fan()
    generate_aba_trilayer_bandstructure()


if __name__ == "__main__":
    main()


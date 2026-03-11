# CHGNetInterface.jl

[![Build Status](https://github.com/cometscome/CHGNetInterface.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cometscome/CHGNetInterface.jl/actions/workflows/CI.yml?query=branch%3Amain)


Julia interface for the CHGNet machine-learning interatomic potential.

`CHGNetInterface.jl` provides a lightweight Julia wrapper around the official
Python implementation of CHGNet so that pretrained or user-trained CHGNet
models can be evaluated directly from Julia.

The package focuses on energy, force, stress, virial, and magnetic-moment
evaluation and is intended to integrate CHGNet potentials into Julia simulation
workflows. Python dependencies are managed automatically using `CondaPkg.jl`.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/cometscome/CHGNetInterface.jl")
```

## Quick Example

```julia
using CHGNetInterface

symbols = ["O", "H", "H"]
positions = [
    0.000000  0.000000  0.000000
    0.758602  0.000000  0.504284
   -0.758602  0.000000  0.504284
]
cell = [
    10.0 0.0 0.0
    0.0 10.0 0.0
    0.0 0.0 10.0
]

pot = CHGNetPotential(
    symbols,
    positions;
    cell=cell,
    pbc=(true, true, true),
    device="cpu",
    model_name="0.3.0",
)

energy(pot)
forces(pot)
stress(pot)
virial(pot)
magmoms(pot)
```

## Notes

- `stress(pot)` follows ASE calculator conventions and returns stress in
  `eV/Angstrom^3`.
- `virial(pot)` is computed from the stress tensor as `-V * σ`.
- `site_energies(pot)` requires `return_site_energies=true` at construction.

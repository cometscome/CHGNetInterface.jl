module CHGNetInterface

using PythonCall

export CHGNetPotential,
    set_positions!, set_cell!, set_pbc!,
    energy, forces, stress, virial, magmoms, site_energies,
    volume, natoms, cell, pbc,
    energy_forces, energy_forces_stress, energy_forces_virial,
    unit_system, energy_unit, force_unit, stress_unit, virial_unit,
    model_version, n_params

const _ase_atoms = Ref{Py}()
const _chgnet_model = Ref{Py}()
const _chgnet_dyn = Ref{Py}()
const _np = Ref{Py}()

function __init__()
    _ase_atoms[] = pyimport("ase.atoms")
    _chgnet_model[] = pyimport("chgnet.model.model")
    _chgnet_dyn[] = pyimport("chgnet.model.dynamics")
    _np[] = pyimport("numpy")
end

mutable struct CHGNetPotential
    calc::Py
    atoms::Py
    natoms::Int
end

"""
    CHGNetPotential(symbols, positions; kwargs...)
    CHGNetPotential(model_path, symbols, positions; kwargs...)

Create a CHGNet calculator and an ASE `Atoms` object, and keep them alive for
repeated evaluations in the current Julia session.

Constructors
------------
- `CHGNetPotential(symbols, positions; ...)`
  Load a pretrained CHGNet model via `CHGNet.load(...)`.
- `CHGNetPotential(model_path, symbols, positions; ...)`
  Load a user model file via `CHGNetCalculator.from_file(...)`.

Arguments
---------
- `model_path`: path to a trained CHGNet model file
- `symbols`: vector of chemical symbols
- `positions`: `natoms × 3` matrix

Keyword arguments
-----------------
- `cell`: optional `3 × 3` cell matrix
- `pbc`: optional periodic boundary condition flags, e.g. `(true, true, true)`
- `device`: `"cpu"`, `"cuda"`, or `"mps"`
- `model_name`: pretrained CHGNet checkpoint name, e.g. `"0.3.0"` or `"r2scan"`
- `check_cuda_mem`: passed to `CHGNetCalculator`
- `on_isolated_atoms`: one of `"ignore"`, `"warn"`, `"error"`
- `return_site_energies`: whether site energies should be returned by the calculator
"""
function CHGNetPotential(
    symbols::Vector{String},
    positions::AbstractMatrix{<:Real};
    cell::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    pbc::Union{Nothing,NTuple{3,Bool}}=nothing,
    device::String="cpu",
    model_name::String="0.3.0",
    check_cuda_mem::Bool=false,
    on_isolated_atoms::String="warn",
    return_site_energies::Bool=false,
)
    natoms_ = length(symbols)
    _check_positions(positions, natoms_)
    _check_isolated_atom_mode(on_isolated_atoms)

    model = _chgnet_model[].CHGNet.load(model_name=model_name, verbose=false, use_device=device)
    calc = _chgnet_dyn[].CHGNetCalculator(
        model=model,
        use_device=device,
        check_cuda_mem=check_cuda_mem,
        on_isolated_atoms=on_isolated_atoms,
        return_site_energies=return_site_energies,
    )
    atoms = _build_atoms(symbols, positions; cell=cell, pbc=pbc)
    atoms.calc = calc
    return CHGNetPotential(calc, atoms, natoms_)
end

function CHGNetPotential(
    model_path::AbstractString,
    symbols::Vector{String},
    positions::AbstractMatrix{<:Real};
    cell::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    pbc::Union{Nothing,NTuple{3,Bool}}=nothing,
    device::String="cpu",
    check_cuda_mem::Bool=false,
    on_isolated_atoms::String="warn",
    return_site_energies::Bool=false,
)
    isfile(model_path) || throw(ArgumentError("model file not found: $model_path"))
    natoms_ = length(symbols)
    _check_positions(positions, natoms_)
    _check_isolated_atom_mode(on_isolated_atoms)

    calc = _chgnet_dyn[].CHGNetCalculator.from_file(
        model_path,
        use_device=device,
        check_cuda_mem=check_cuda_mem,
        on_isolated_atoms=on_isolated_atoms,
        return_site_energies=return_site_energies,
    )
    atoms = _build_atoms(symbols, positions; cell=cell, pbc=pbc)
    atoms.calc = calc
    return CHGNetPotential(calc, atoms, natoms_)
end

# ----------------------------
# Internal helpers
# ----------------------------

function _to_numpy_matrix(x::AbstractMatrix{<:Real})
    return _np[].array(Matrix{Float64}(x), dtype=_np[].float64)
end

function _build_atoms(
    symbols::Vector{String},
    positions::AbstractMatrix{<:Real};
    cell::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
    pbc::Union{Nothing,NTuple{3,Bool}}=nothing,
)
    kwargs = Dict{Symbol,Any}()
    kwargs[:symbols] = symbols
    kwargs[:positions] = _to_numpy_matrix(positions)
    if cell !== nothing
        _check_cell(cell)
        kwargs[:cell] = _to_numpy_matrix(cell)
    end
    if pbc !== nothing
        _check_pbc_flags(pbc)
        kwargs[:pbc] = collect(pbc)
    end
    return _ase_atoms[].Atoms(; kwargs...)
end

function _check_positions(positions::AbstractMatrix, natoms_::Integer)
    size(positions, 1) == natoms_ ||
        throw(ArgumentError("positions must have size ($(natoms_), 3), got $(size(positions))"))
    size(positions, 2) == 3 ||
        throw(ArgumentError("positions must have size ($(natoms_), 3), got $(size(positions))"))
    return nothing
end

function _check_cell(cell::AbstractMatrix)
    size(cell, 1) == 3 && size(cell, 2) == 3 ||
        throw(ArgumentError("cell must have size (3, 3), got $(size(cell))"))
    return nothing
end

_check_pbc_flags(::NTuple{3,Bool}) = nothing

function _check_isolated_atom_mode(mode::AbstractString)
    mode in ("ignore", "warn", "error") ||
        throw(ArgumentError("on_isolated_atoms must be one of \"ignore\", \"warn\", \"error\"; got \"$mode\""))
    return nothing
end

# ----------------------------
# Structure mutators
# ----------------------------

"""
    set_positions!(pot, positions)

Update atomic positions in the stored ASE `Atoms` object.
"""
function set_positions!(pot::CHGNetPotential, positions::AbstractMatrix{<:Real})
    _check_positions(positions, pot.natoms)
    pot.atoms.positions = _to_numpy_matrix(positions)
    return pot
end

"""
    set_cell!(pot, cell; scale_atoms=false)

Update the simulation cell. If `scale_atoms=true`, ASE rescales atom positions
together with the cell.
"""
function set_cell!(
    pot::CHGNetPotential,
    newcell::AbstractMatrix{<:Real};
    scale_atoms::Bool=false,
)
    _check_cell(newcell)
    pot.atoms.set_cell(_to_numpy_matrix(newcell), scale_atoms=scale_atoms)
    return pot
end

"""
    set_pbc!(pot, pbc)

Update periodic boundary condition flags. Example: `(true, true, true)`.
"""
function set_pbc!(pot::CHGNetPotential, newpbc::NTuple{3,Bool})
    _check_pbc_flags(newpbc)
    pot.atoms.pbc = collect(newpbc)
    return pot
end

# ----------------------------
# Structure accessors
# ----------------------------

"""
    natoms(pot)

Number of atoms.
"""
natoms(pot::CHGNetPotential) = pot.natoms

"""
    cell(pot)

Return the current cell as a `3 × 3` Julia matrix.
"""
function cell(pot::CHGNetPotential)
    return pyconvert(Matrix{Float64}, pot.atoms.cell.array)
end

"""
    pbc(pot)

Return the periodic boundary condition flags as a 3-tuple of `Bool`.
"""
function pbc(pot::CHGNetPotential)
    flags = pyconvert(Vector{Bool}, pot.atoms.pbc)
    length(flags) == 3 || throw(ErrorException("unexpected pbc length: $(length(flags))"))
    return (flags[1], flags[2], flags[3])
end

"""
    volume(pot)

Return the cell volume. This requires that a valid cell is defined in the stored
ASE `Atoms` object.
"""
function volume(pot::CHGNetPotential)
    return pyconvert(Float64, pot.atoms.get_volume())
end

"""
    model_version(pot)

Return the CHGNet model version string when available.
"""
function model_version(pot::CHGNetPotential)
    v = pot.calc.version
    return pyconvert(Union{Nothing,String}, v)
end

"""
    n_params(pot)

Return the number of model parameters.
"""
function n_params(pot::CHGNetPotential)
    return pyconvert(Int, pot.calc.n_params)
end

# ----------------------------
# Energy / forces / stress / virial / magnetic moments
# ----------------------------

"""
    energy(pot)

Potential energy in eV.
"""
function energy(pot::CHGNetPotential)
    return pyconvert(Float64, pot.atoms.get_potential_energy())
end

"""
    forces(pot)

Atomic forces as a `natoms × 3` Julia matrix in eV/Å.
"""
function forces(pot::CHGNetPotential)
    return pyconvert(Matrix{Float64}, pot.atoms.get_forces())
end

"""
    stress(pot; voigt=false)

Configuration stress.
- `voigt=false`: returns a `3 × 3` matrix
- `voigt=true`: returns a 6-component Voigt vector

The returned units follow ASE calculator conventions, i.e. eV/Å^3.
"""
function stress(pot::CHGNetPotential; voigt::Bool=false)
    s = pot.atoms.get_stress(voigt=voigt)
    if voigt
        return pyconvert(Vector{Float64}, s)
    else
        return pyconvert(Matrix{Float64}, s)
    end
end

"""
    virial(pot)

Configuration virial tensor as a `3 × 3` matrix in eV.
Computed from the stress tensor by `W = -V * σ`, where `V` is the cell volume
and `σ` is the configuration stress.
"""
function virial(pot::CHGNetPotential)
    σ = stress(pot; voigt=false)
    V = volume(pot)
    return -V .* σ
end

"""
    magmoms(pot)

Return atomic magnetic moments as a vector in `μ_B`.
"""
function magmoms(pot::CHGNetPotential)
    return pyconvert(Vector{Float64}, pot.atoms.get_magnetic_moments())
end

"""
    site_energies(pot)

Return site energies in eV if `return_site_energies=true` was enabled when the
calculator was constructed.
"""
function site_energies(pot::CHGNetPotential)
    # Force a calculation if needed.
    _ = energy(pot)
    if !pyhasattr(pot.calc, "results") || !haskey(PyDict(pot.calc.results), "energies")
        throw(ArgumentError("site energies are unavailable; construct CHGNetPotential with return_site_energies=true"))
    end
    return pyconvert(Vector{Float64}, pot.calc.results["energies"])
end

# ----------------------------
# Combined evaluation helpers
# ----------------------------

"""
    energy_forces(pot, positions)

Update positions and return `(energy, forces)`.
"""
function energy_forces(pot::CHGNetPotential, positions::AbstractMatrix{<:Real})
    set_positions!(pot, positions)
    return energy(pot), forces(pot)
end

"""
    energy_forces_stress(pot, positions)

Update positions and return `(energy, forces, stress)`.
"""
function energy_forces_stress(pot::CHGNetPotential, positions::AbstractMatrix{<:Real})
    set_positions!(pot, positions)
    return energy(pot), forces(pot), stress(pot; voigt=false)
end

"""
    energy_forces_virial(pot, positions)

Update positions and return `(energy, forces, virial)`.
"""
function energy_forces_virial(pot::CHGNetPotential, positions::AbstractMatrix{<:Real})
    set_positions!(pot, positions)
    return energy(pot), forces(pot), virial(pot)
end

# ----------------------------
# Unit-system helpers
# ----------------------------

"""
    unit_system()

Return a `NamedTuple` describing the unit convention used by this interface.
This interface follows ASE calculator conventions:
- energy: eV
- length: Å
- force: eV/Å
- stress: eV/Å^3
- virial: eV
- magnetic moment: μ_B
"""
function unit_system()
    return (
        energy="eV",
        length="Angstrom",
        force="eV/Angstrom",
        stress="eV/Angstrom^3",
        virial="eV",
        magnetic_moment="mu_B",
    )
end

"""Return the energy unit string."""
energy_unit() = unit_system().energy

"""Return the force unit string."""
force_unit() = unit_system().force

"""Return the stress unit string."""
stress_unit() = unit_system().stress

"""Return the virial unit string."""
virial_unit() = unit_system().virial

end

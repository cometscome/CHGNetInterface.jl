using Test
using CHGNetInterface
using PythonCall

const TEST_SYMBOLS = ["O", "H", "H"]
const TEST_POSITIONS = [
     0.000000  0.000000  0.000000
     0.758602  0.000000  0.504284
    -0.758602  0.000000  0.504284
]
const TEST_CELL = [
    10.0 0.0 0.0
    0.0 10.0 0.0
    0.0 0.0 10.0
]

"""
Return a CHGNetPotential for tests.

Priority:
1. CHGNET_TEST_MODEL if it points to an existing file.
2. test/test_model.pt if present.
3. Otherwise use a pretrained model name and let CHGNet download/cache it.
"""
function make_test_potential(; return_site_energies::Bool=false)
    model_path = get(ENV, "CHGNET_TEST_MODEL", joinpath(@__DIR__, "test_model.pt"))
    if isfile(model_path)
        @info "Using local CHGNet test model" model_path
        return CHGNetPotential(
            model_path,
            TEST_SYMBOLS,
            TEST_POSITIONS;
            cell=TEST_CELL,
            pbc=(true, true, true),
            device="cpu",
            return_site_energies=return_site_energies,
        )
    else
        model_name = get(ENV, "CHGNET_TEST_MODEL_NAME", "0.3.0")
        @info "Using pretrained CHGNet test model (will download if needed)" model_name
        return CHGNetPotential(
            TEST_SYMBOLS,
            TEST_POSITIONS;
            cell=TEST_CELL,
            pbc=(true, true, true),
            model_name=model_name,
            device="cpu",
            return_site_energies=return_site_energies,
        )
    end
end

@testset "CHGNetInterface.jl" begin
    @testset "Imports" begin
        @test !isnothing(pyimport("numpy"))
        @test !isnothing(pyimport("ase.atoms"))
        @test !isnothing(pyimport("chgnet.model.model"))
        @test !isnothing(pyimport("chgnet.model.dynamics"))
    end

    @testset "Unit system" begin
        u = unit_system()
        @test u.energy == "eV"
        @test u.length == "Angstrom"
        @test u.force == "eV/Angstrom"
        @test u.stress == "eV/Angstrom^3"
        @test u.virial == "eV"
        @test energy_unit() == "eV"
        @test force_unit() == "eV/Angstrom"
        @test stress_unit() == "eV/Angstrom^3"
        @test virial_unit() == "eV"
    end

    @testset "Constructor validation" begin
        @test_throws ArgumentError CHGNetPotential(
            "no_such_file.pt",
            ["H"],
            zeros(1, 3),
        )

        try
            pot = make_test_potential()
            @test_throws ArgumentError set_positions!(pot, zeros(2, 3))
            @test_throws ArgumentError set_positions!(pot, zeros(3, 2))
            @test_throws ArgumentError set_cell!(pot, zeros(2, 2))
            @test_throws ArgumentError set_cell!(pot, zeros(3, 2))
        catch e
            @info "Skipping shape-validation tests because CHGNet model setup/download failed." exception=(typeof(e), sprint(showerror, e))
        end
    end

    @testset "Integration" begin
        pot = nothing
        setup_ok = false
        try
            pot = make_test_potential(return_site_energies=true)
            setup_ok = true
        catch e
            @info "Skipping CHGNet integration tests because pretrained model setup/download failed." exception=(typeof(e), sprint(showerror, e))
        end

        if setup_ok
            @test natoms(pot) == 3
            c0 = cell(pot)
            @test size(c0) == (3, 3)
            @test isapprox(c0, TEST_CELL; atol=1e-12)
            @test pbc(pot) == (true, true, true)

            V0 = volume(pot)
            @test V0 isa Float64
            @test isfinite(V0)
            @test isapprox(V0, 1000.0; atol=1e-10)

            e0 = energy(pot)
            f0 = forces(pot)
            s0 = stress(pot)
            s0v = stress(pot; voigt=true)
            w0 = virial(pot)
            m0 = magmoms(pot)
            se0 = site_energies(pot)

            @test e0 isa Float64
            @test isfinite(e0)
            @test size(f0) == (3, 3)
            @test eltype(f0) == Float64
            @test all(isfinite, f0)
            @test size(s0) == (3, 3)
            @test eltype(s0) == Float64
            @test all(isfinite, s0)
            @test length(s0v) == 6
            @test eltype(s0v) == Float64
            @test all(isfinite, s0v)
            @test size(w0) == (3, 3)
            @test eltype(w0) == Float64
            @test all(isfinite, w0)
            @test isapprox(w0, -V0 .* s0; atol=1e-10)
            @test length(m0) == 3
            @test eltype(m0) == Float64
            @test all(isfinite, m0)
            @test length(se0) == 3
            @test eltype(se0) == Float64
            @test all(isfinite, se0)

            @testset "Position updates" begin
                x1 = copy(TEST_POSITIONS)
                x1[2, 3] += 0.01
                set_positions!(pot, x1)
                e1 = energy(pot)
                f1 = forces(pot)
                @test e1 isa Float64
                @test size(f1) == (3, 3)
                @test all(isfinite, f1)
                @test !(e0 == e1 && f0 == f1)
            end

            @testset "Combined evaluation APIs" begin
                x2 = copy(TEST_POSITIONS)
                x2[1, 1] += 0.02

                e2, f2 = energy_forces(pot, x2)
                @test e2 isa Float64
                @test size(f2) == (3, 3)
                @test all(isfinite, f2)

                e3, f3, s3 = energy_forces_stress(pot, x2)
                @test e3 isa Float64
                @test size(f3) == (3, 3)
                @test size(s3) == (3, 3)
                @test all(isfinite, f3)
                @test all(isfinite, s3)

                e4, f4, w4 = energy_forces_virial(pot, x2)
                @test e4 isa Float64
                @test size(f4) == (3, 3)
                @test size(w4) == (3, 3)
                @test all(isfinite, f4)
                @test all(isfinite, w4)
                @test isapprox(w4, -volume(pot) .* stress(pot); atol=1e-10)
            end

            @testset "Cell updates" begin
                newcell = [
                    11.0 0.0 0.0
                    0.0 10.0 0.0
                    0.0 0.0 10.0
                ]
                set_cell!(pot, newcell; scale_atoms=false)
                c1 = cell(pot)
                @test size(c1) == (3, 3)
                @test isapprox(c1, newcell; atol=1e-12)
                V1 = volume(pot)
                @test isfinite(V1)
                @test isapprox(V1, 1100.0; atol=1e-10)
            end

            @testset "PBC updates" begin
                set_pbc!(pot, (true, false, true))
                @test pbc(pot) == (true, false, true)
                set_pbc!(pot, (false, false, false))
                @test pbc(pot) == (false, false, false)
                set_pbc!(pot, (true, true, true))
                @test pbc(pot) == (true, true, true)
            end
        end
    end
end

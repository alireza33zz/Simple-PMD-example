"""
This script performs a three-phase optimal power flow (OPF) calculation using the PowerModelsDistribution package in Julia. 
"""

using PowerModelsDistribution
using JuMP
using Ipopt
using DataFrames

"""
Configuration struct to hold OPF parameters
"""
struct OPFConfig
    sbase_default::Float64
    power_scale_factor::Float64
    v_upper_bound::Float64
    v_lower_bound::Float64
    print_level::Int
end

"""
Default configuration
"""
function default_config()
    return OPFConfig(
        1.0,    # sbase_default
        1000.0, # power_scale_factor
        1.10,   # v_upper_bound
        0.94,   # v_lower_bound
        1       # print_level
    )
end

"""
Custom constraint type for extensibility
"""
abstract type AbstractConstraint end

struct VoltageConstraint <: AbstractConstraint
    v_upper_bound::Float64
    v_lower_bound::Float64
end

"""
Add voltage magnitude constraints to the model
"""
function add_voltage_constraints!(pm, constraint::VoltageConstraint)
    pm.model[:voltage_constraints] = Dict()
    for i in 1:length(pm.data["bus"])
        for phase in 1:3
            con = @constraint(pm.model, constraint.v_lower_bound <= pm.var[:it][:pmd][:nw][0][:vm][i][phase] <= constraint.v_upper_bound)
            pm.model[:voltage_constraints][(i, phase)] = con
        end
    end
    println("")
    printstyled("Voltage magnitude constraints integrated successfully!"; color=:yellow)
end

"""
Configure solver with custom settings
"""
function configure_solver(config::OPFConfig)
    return JuMP.optimizer_with_attributes(
        Ipopt.Optimizer,
        "print_level" => config.print_level,
        "tol" => 1e-8,  # Adjust tolerance
        "acceptable_tol" => 1e-8  # Adjust acceptable tolerance
    )
end

"""
Initialize PowerModelsDistribution model with basic settings
"""
function initialize_model(eng::Dict, config::OPFConfig)
    eng["settings"]["sbase_default"] = config.sbase_default
    eng["settings"]["power_scale_factor"] = config.power_scale_factor
    math = transform_data_model(eng)

    # Add generator cost coefficients
    math["gen"]["1"]["cost"] = [1.0, 0.0]

    pm = instantiate_mc_model(math, ACPUPowerModel, build_mc_opf)
    pm.data["per_unit"] = false
    
    return pm, math
end

"""
Extract and format key results from the OPF solution with VUF
"""
function format_results(solution::Dict, pm)
    # Initialize DataFrame for bus results
    results_df = DataFrame(
        bus_id = String[],
        phase = String[],  # Change to String[]
        vm_pu = Float64[],
        va_deg = Float64[]
    )

    #  Define phase mapping
    phase_map = Dict(1 => "a", 2 => "b", 3 => "c")

    # Extract bus results
    for (bus_id, bus) in solution["solution"]["bus"]
        # Collect voltage magnitudes and angles for this bus
        vm = bus["vm"]
        va = bus["va"] * (180/π)  # Convert to degrees for readability
       
        # Add rows to DataFrame
        for phase in 1:3
            push!(results_df, (
                bus_id,
                phase_map[phase],  # Use the phase mapping
                round(vm[phase], digits=3),
                round(va[phase], digits=1)
            ))
        end
    end
    
    # Sort by bus_id and phase
    sort!(results_df, [:bus_id, :phase])
    
    return results_df
end

"""
Print formatted results with VUF
"""
function print_formatted_results(solution::Dict, pm)
    if solution["termination_status"] != MOI.LOCALLY_SOLVED
        println()
        printstyled("WARNING: OPF didn't converge!"; color=:red)
        println()
        return
    end

    # Format and display results
    results_df = format_results(solution, pm)
    printstyled("\n=== OPF Results ==="; color = :green)
    println()
    printstyled("Objective value: ", round(solution["objective"], digits=3); color = :blue)
    println()
    println("Solve time: ", round(solution["solve_time"], digits=6), " seconds")
    println("\nBus Results:")
    println(results_df)
end

"""
Main function to solve OPF with custom constraints
"""
function solve_opf(file_path::String, config::OPFConfig=default_config())

    # Parse network file
    eng = parse_file(file_path)

    # Initialize model
    pm, math = initialize_model(eng, config)

    # Add voltage magnitude constraints
    voltage_constraint = VoltageConstraint(config.v_upper_bound, config.v_lower_bound)
    add_voltage_constraints!(pm, voltage_constraint)

    # Configure and run solver
    solver = configure_solver(config)
    solution = optimize_model!(pm, optimizer=solver)

    # Print formatted results
    print_formatted_results(solution, pm)

    return solution, pm, format_results(solution, pm)
end

"""
Test function with default settings
"""
function test_opf_with_vuf(file_path::String)
    # Define custom configuration
    config = OPFConfig(
        1.0,    # sbase_default
        1000.0, # power_scale_factor
        1.10,   # v_upper_bound
        0.94,   # v_lower_bound
        1       # print_level
    )
   
    # Solve OPF with VUF constraint
    solution, pm, results_df = solve_opf(file_path, config)
    
    return solution, pm, results_df
end

# Example usage:
file_path = "Test1.dss"
solution, pm, results_df = test_opf_with_vuf(file_path);

# Optional: Diagnostic function to understand network structure
function print_network_structure(solution::Dict)
    println("\n=================================================================================")
    println("Complete report:\n")

    println("Buses:")
    for (bus_id, bus) in solution["solution"]["bus"]
        println("Bus $bus_id")
    end
    
    println("\nGenerators:")
    for (gen_id, gen) in solution["solution"]["gen"]
        println("Generator $gen_id:")
        println("  pg_bus: ", round.(get(gen, "pg_bus", "Not found"), digits=4))
        println("  qg_bus: ", round.(get(gen, "qg_bus", "Not found"), digits=4))
    end
    
    println("\nLoads:")
    for (load_id, load) in solution["solution"]["load"]
        println("Load $load_id:")
        println("  pd_bus: ", round.(get(load, "pd_bus", "Not found"), digits=4))
        println("  qd_bus: ", round.(get(load, "qd_bus", "Not found"), digits=4))
    end
end

print_network_structure(solution);



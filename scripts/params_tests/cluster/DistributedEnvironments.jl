module DistributedEnvironments
# check https://github.com/albheim/DistributedEnvironments.jl/tree/main for original implementation
export @initcluster, @eachmachine, @everywhere, cluster_status

using Distributed, Pkg, MacroTools


"""
    @initcluster ips [worker_procs=:auto] [sync=true] [status=false]

Takes a list of ip-strings and sets up the current environment on these machines.
The machines should be reachable using ssh from the machine running the command.
The setup will copy the local project and manifest files as well as copying all
packages that are added to the env using `dev` to corresponding location on the 
remote machines.

Additional arguments:
* `worker_procs` - Integer or :auto (default), how many workers are added on each machine.
* `sync` - Whether or not to sync the local environment (default is `true`) before adding the workers.
* `status` - Whether or not to show a short status (current users, cpu utilization, julia version) for each machine and remove any machine that does not connect. Default is `false`.

Needs to be called from top level since the macro includes imports.

# Example
```julia
using DistributedEnvironments

ips = ["10.0.0.1", "10.0.0.2"]
@initcluster ips

@everywhere using SomePackage
...
"""

function initcluster(cluster_config_main::Dict, cluster_config_hosts::Vector{<:Dict}; sync=true)
    if length(Distributed.procs()) > 1
        workers_to_remove_ids = [pid for pid in Distributed.workers() if pid != 1]
        Distributed.rmprocs(workers_to_remove_ids...)
        println("Removed workers $workers_to_remove_ids")
    end

    cluster_status!(cluster_config_hosts)

    printstyled("Adding main host -> $(cluster_config_main[:use_n_workers]) workers\n", bold=true, color=:magenta)
    env = Dict(
        "JULIA_NUM_THREADS" => "$(cluster_config_main[:julia_threads_per_worker])",
        "JULIA_BLAS_THREADS" => "$(cluster_config_main[:blas_threads_per_worker])"
    )
    enable_threaded_blas = cluster_config_main[:blas_threads_per_worker] > 1
    Distributed.addprocs(
        cluster_config_main[:use_n_workers],
        env=env,
        enable_threaded_blas=enable_threaded_blas,
        topology=:master_worker
    )
    println("Added main host\n")

    # Sync and instantiate (does precompilation)
    # if $(sync)
    #     # Sync local packages and environment files to all nodes
    #     sync_env(cluster)

    #     # Add single worker on each machine to precompile
    #     addprocs(
    #         map(node -> (node, 1), cluster), 
    #         topology = :master_worker, 
    #         tunnel = true, 
    #         exeflags = "--project=$(Base.active_project())",
    #         max_parallel = length(cluster), 
    #     ) 

    #     # Instantiate environment on all machines
    #     @everywhere begin
    #         import Pkg
    #         Pkg.instantiate()
    #     end

    #     # Remove precompile workers
    #     # TODO should be able to keep them and add the right amount, maybe SSHManager?
    #     # Or maybe not, good to have restart of all machines after precompile since sometimes it hangs there.
    #     rmprocs(workers())
    # end

    # # Add one worker per thread on each node in the cluster
    # addprocs(
    #     map(node -> (node, $(worker_procs)), cluster), 
    #     topology=:master_worker, 
    #     tunnel=true, 
    #     exeflags = "--project=$(Base.active_project())",
    #     max_parallel=24*length(cluster), # TODO what should this be?
    # ) 
    # println("All workers initialized.")
end

"""
    @eachmachine expr

Similar to `@everywhere`, but only runs on one worker per machine.
Can be used for things like precompiling, downloading datasets or similar.
"""
macro eachmachine(expr)
    return _eachmachine(expr)
end

function _eachmachine(expr)
    machinepids = get_unique_machine_ids()
    quote 
        @everywhere $machinepids $expr
    end
end

get_unique_machine_ids() = unique(id -> Distributed.get_bind_addr(id), procs())
get_unique_machine_ips() = unique(map(id -> Distributed.get_bind_addr(id), procs()))

function sync_env(cluster)
    proj_path = dirname(Pkg.project().path)
    deps = Pkg.dependencies()
    # (:name, :version, :tree_hash, :is_direct_dep, :is_pinned, :is_tracking_path, :is_tracking_repo, :is_tracking_registry, :git_revision, :git_source, :source, :dependencies)

    Threads.@threads for node in cluster
        rsync("$(proj_path)/Project.toml", node)
        println("Worker $(node): Copied Project.toml")
        rsync("$(proj_path)/Manifest.toml", node)
        println("Worker $(node): Copied Manifest.toml")

        for (id, package) in deps
            if package.is_tracking_path
                rsync(package.source, node)
                println("Worker $(node): Copied $(package.name)")
            end
        end
    end
end

function rsync(path, target)
    run(`ssh -q -t $(target) mkdir -p $(dirname(path))`) # Make sure path exists
    if isfile(path)
        run(`rsync -ue ssh $(path) $(target):$(path)`) # Copy
    else
        run(`rsync -rue ssh --delete $(path)/ $(target):$(path)`) # Copy
    end
end

function scp(path, target)
    dir = isfile(path) ? dirname(path) : path
    run(`ssh -q -t $(target) mkdir -p $(dir)`) # Make sure path exists
    run(`ssh -q -t $(target) rm -rf $(path)`) # Delete old
    run(`scp -r -q $(path) $(target):$(path)`) # Copy
end

"""
    cluster_status!(cluster)

Run a status check on each machine in the list and throws an error if any of them is unreachable.
"""
function cluster_status!(cluster::Vector{<:Dict})
    printstyled("\n\n\nChecking machine main:\n", bold=true, color=:magenta)
    output = String(read(`julia -e "print(VERSION)"`))
    output = last(split(output, "\n", keepempty=false))
    # println(output)

    connection_error = []
    for node in cluster
        printstyled("\nChecking machine $(node[:host_address]):\n", bold=true, color=:magenta)
        try
            output = String(read(`ssh -i $(node[:private_key_path]) -p $(node[:port]) -t $(node[:host_address]) $(node[:exe_path]) -e "print(VERSION)"`))
            output = last(split(output, "\n", keepempty=false))
            println(output)
        catch e
            connection_error = vcat(connection_error, (node, e))
        end
    end

    # filter!(x -> !(x in connection_error), cluster)

    if !isempty(connection_error)
        println("Failed to connect to the following machines:") 
        for (node, e) in connection_error
            println("\t $(node[:host_address])   $(e)")
        end
        throw("Failed connect to one of machines, you should check in manually and change / remove these hosts")
    end

    printstyled("\n\nAll machines are reachable\n", bold=true, color=:magenta)
end

end
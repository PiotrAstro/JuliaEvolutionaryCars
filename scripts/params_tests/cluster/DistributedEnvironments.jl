# check https://github.com/albheim/DistributedEnvironments.jl/tree/main for original implementation
module DistributedEnvironments

import Pkg
using Distributed, MacroTools

export @eachmachine, @eachworker, @initcluster, cluster_status!, remove_workers!, addprocs_one_host!, host_status, hosts_manager

macro initcluster(cluster_main, cluster_hosts, tmp_dir_name, copy_env_and_code)
    return _initcluster(cluster_main, cluster_hosts, tmp_dir_name, copy_env_and_code)
end

"""
Function to add workers to the cluster.
It will automatically set up workers.
It will copy code and project info and precompile them, if copy_env_and_code is set to true.
Otherwise it will not do it, it will just go to the right place and activate env, without instantiiate it, it should only be used if one already copied everything and it is all set up.
"""
function _initcluster(
    cluster_config_main_macro_input,
    cluster_config_hosts_main_macro_input,
    tmp_dir_name_macro_input,
    copy_env_and_code_main_macro_input,
)
    quote
        cluster_config_main = $(esc(cluster_config_main_macro_input))
        cluster_config_hosts = $(esc(cluster_config_hosts_main_macro_input))
        copy_env_and_code = $(esc(copy_env_and_code_main_macro_input))
        tmp_dir_name = $(esc(tmp_dir_name_macro_input))

        added_pids = []

        remove_workers!()

        cluster_status!(cluster_config_hosts)
        for host in cluster_config_hosts
            host[:dir_project] = joinpath(host[:dir], tmp_dir_name)
        end

        if copy_env_and_code
            println("Copying project to all machines")
            project_archive = "_tmp_julia_comp_$(tmp_dir_name).tar.gz"
            create_project_archive(project_archive)
            try
                for host in cluster_config_hosts
                    printstyled("Copying project to $(host[:host_address]) : $(host[:dir_project])\n", bold=true, color=:magenta)
                    if host[:shell] == :wincmd
                        run(`ssh -i $(host[:private_key_path]) $(host[:host_address]) "(if exist $(host[:dir_project]) (rd /q /s $(host[:dir_project]))) & mkdir $(host[:dir_project])"`)
                        println("Created dir and removed old if existed $(host[:dir_project]), copying project...")
                        dir_project_slash = host[:dir_project] * "\\"
                        run(`scp -i $(host[:private_key_path]) $project_archive $(host[:host_address]):$dir_project_slash`)
                        println("Copied project, extracting...")
                        run(`ssh -i $(host[:private_key_path]) $(host[:host_address]) "tar -xzf $dir_project_slash$project_archive -C $(host[:dir_project])"`)
                        println("Extracted project")
                    else
                        throw("other shells than :wincmd are currently not implemented, implement them yourself!")
                    end
                    println("Adding one worker to $(host[:host_address]) for setup...")
                    addprocs_one_host!(host, 1)
                    println("Worker added.")
                end
                rm(project_archive)

                printstyled("\nAll projects initialized. Precompiling...\n\n", bold=true, color=:green)
                @everywhere begin
                    import Pkg
                    println("Instantiating project: " * Pkg.project().name)
                    Pkg.instantiate()
                end
                printstyled("\nAll hosts precompiled.\n\n", bold=true, color=:green)

                # we will add other workers again
                rmprocs(workers())
            catch e
                println("Failed to copy project to all machines, you should check in manually and change / remove these hosts")
                throw(e)
            finally
                if isfile(project_archive)
                    rm(project_archive)
                end
            end
        else
            println("Skipping copying and instatiating project to all machines")
        end

        try
            printstyled("Adding main host -> $(cluster_config_main[:use_n_workers]) workers\n", bold=true, color=:magenta)
            Distributed.addprocs(
                cluster_config_main[:use_n_workers],
                env=["JULIA_BLAS_THREADS" => "$(cluster_config_main[:blas_threads_per_worker])"],
                exeflags="--threads=$(cluster_config_main[:julia_threads_per_worker])",
                enable_threaded_blas=cluster_config_main[:blas_threads_per_worker] > 1,
                topology=:master_worker
            )
            println("Added main host\n")

            for host in cluster_config_hosts
                printstyled("Adding $(host[:host_address]) -> $(host[:use_n_workers]) workers\n", bold=true, color=:magenta)
                addprocs_one_host!(host)
                println("Added $(host[:host_address])\n")
            end

            printstyled("All workers added\n", bold=true, color=:green)
        catch e
            println("Failed adding some workers, you should check it manually and change / remove these hosts")
            throw(e)
        end
    end
end

function remove_workers!()
    if length(Distributed.procs()) > 1
        workers_to_remove_ids = [pid for pid in Distributed.workers() if pid != 1]
        Distributed.rmprocs(workers_to_remove_ids...)
        println("Removed workers $workers_to_remove_ids")
    else
        println("No workers to remove")
    end
end 

"""
    @eachmachine expr

Similar to `@everywhere`, but only runs on one worker per machine.
Can be used for things like precompiling, downloading datasets or similar.
"""
macro eachmachine(expr)
    return _eachmachine(expr)
end

"""
    @eachworker expr

Similar to `@everywhere`, but runs only on workers, not on main => not on pid 1.
"""
macro eachworker(expr)
    return _eachworker(expr)
end

function _eachworker(expr)
    workerspids = workers()
    quote
        @everywhere $workerspids $expr
    end
end

function _eachmachine(expr)
    machinepids = get_unique_machine_ids()
    quote 
        @everywhere $machinepids $expr
    end
end

function addprocs_one_host!(host::Dict, workers_n=-1) # if workers_n = -1, it will take workers_n from host
    added_pids = Distributed.addprocs(
        [(host[:host_address], workers_n == -1 ? host[:use_n_workers] : workers_n)],
        sshflags = `-i $(host[:private_key_path])`,
        env = [
            "JULIA_BLAS_THREADS" => "$(host[:blas_threads_per_worker])"
        ],
        shell = host[:shell],
        tunnel = host[:tunnel],
        dir = host[:dir_project],
        # --project=@. is to use the project and manifest in the current directory
        exeflags = ["--project=@.", "--threads=$(host[:julia_threads_per_worker])"], 
        exename = "julia",
        enable_threaded_blas = host[:blas_threads_per_worker] > 1,
        topology = :master_worker,
    )
    host[:pids] = append!(get(host, :pids, Vector{Int}()), added_pids)
end

get_unique_machine_ids() = unique(id -> Distributed.get_bind_addr(id), procs())
get_unique_machine_ips() = unique(map(id -> Distributed.get_bind_addr(id), procs()))


"""
    cluster_status!(cluster)

Run a status check on each machine in the list and throws an error if any of them is unreachable.
"""
function cluster_status!(cluster::Vector)
    printstyled("\n\n\nChecking machine main:\n", bold=true, color=:magenta)
    output = read(`julia -v`, String)
    println("Main host julia version: "*output)

    connection_error = []
    for node in cluster
        result_error, e = host_status(node)
        if !result_error
            push!(connection_error, (node, e))
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

    printstyled("\nAll machines are reachable\n\n", bold=true, color=:green)
end

function host_status(host::Dict)::Tuple{Bool, Union{Nothing, Exception}}
    printstyled("Checking machine $(host[:host_address]):\n", bold=true, color=:magenta)
    try
        output = read(`ssh -i $(host[:private_key_path]) $(host[:host_address]) julia -v`, String)
        println("Host available, julia version: "*output)
        return true, nothing
    catch e
        return false, e
    end
end

function create_project_archive(archive_path::String)
    project_archive = archive_path
    git_absolute_dir = strip(read(`git rev-parse --show-toplevel`, String))
    cd(git_absolute_dir) do
        files = readlines(`git ls-files -c -m -o --exclude-standard`)
        unique!(files)
        existing_files = filter(file -> isfile(file) && file != project_archive, files)
        run(`tar -czf $project_archive $(existing_files)`)
    end
end

function extract_project_archive(archive_path::String, extract_dir::String)
    run(`tar -xzf $archive_path -C $extract_dir`)
end

"""
Currently, it will not work at all, unfortunatelly pmap doesnt handle workers termination, I will have to create my own pmap alternative.

It will take some time, I do not want to do it now, so currently this feature is not used.
"""
function hosts_manager(hosts::Vector{<:Dict}, initializator_function::Function, should_exit::Ref{Bool}, sleep_seconds, check_timeout=20)
    sleep_start_time = time()
    while !should_exit[]
        if time() - sleep_start_time < sleep_seconds
            sleep(1)
            continue
        end
        
        workers = Distributed.workers()
        for host in hosts
            workers_in_pool = [pid in workers for pid in host[:pids]]
            workers_unreachability = map(host[:pids]) do pid
                @async begin
                    try
                        # Simple health check
                        remotecall_fetch(() -> 1, pid; timeout=check_timeout)
                        return false
                    catch
                        return true
                    end
                end
            end
            workers_unreachability = fetch.(workers_unreachability)
            try
                Distributed.rmprocs(workers_in_pool[workers_unreachability])
            catch
                println("Failed to remove dead workers")
            end

            host[:pids] = workers_in_pool[.!workers_unreachability]
            if length(host[:pids]) < host[:use_n_workers]
                println("Some workers died on host $(host[:host_address])")
                result, e = host_status(host)
                if result
                    try
                        pids_n = host[:use_n_workers] - length(host[:pids])
                        new_pids = addprocs_one_host!(host, pids_n)
                        host[:pids] = vcat(host[:pids], new_pids)
                        initializator_function(new_pids)
                    catch
                        println("Failed to add new workers to $(host[:host_address])")
                    end
                else !result
                    println(e)
                end
            else
                println("All workers alive on host $(host[:host_address])")
            end
        end
        sleep_start_time = time()
    end
end

end
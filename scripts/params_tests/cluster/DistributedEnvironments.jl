# check https://github.com/albheim/DistributedEnvironments.jl/tree/main for original implementation
module DistributedEnvironments

import Pkg
using Distributed, MacroTools

export @eachmachine, @eachworker, @initcluster, cluster_status!, remove_workers!

macro initcluster(cluster_main, cluster_hosts, tmp_dir_name, copy_env_and_code)
    return _initcluster(cluster_main, cluster_hosts, tmp_dir_name, copy_env_and_code)
end

"""
Function to add workers to the cluster.
It will automatically set up workers.
It will copy code and project info and precompile them, if copy_env_and_code is set to true.
Otherwise it will not do it, it will just go to the right place and activate env, without instantiiate it, it should only be used if one already copied everything and it is all set up.
"""
function _initcluster(cluster_config_main_macro_input, cluster_config_hosts_main_macro_input, tmp_dir_name_macro_input, copy_env_and_code_main_macro_input)
    quote
        cluster_config_main = $(esc(cluster_config_main_macro_input))
        cluster_config_hosts = $(esc(cluster_config_hosts_main_macro_input))
        copy_env_and_code = $(esc(copy_env_and_code_main_macro_input))
        tmp_dir_name = $(esc(tmp_dir_name_macro_input))

        remove_workers!()

        cluster_status!(cluster_config_hosts)
        for host in cluster_config_hosts
            host[:dir_project] = joinpath(host[:dir], tmp_dir_name)
        end

        if copy_env_and_code
            println("Copying project to all machines")
            project_archive = "_tmp_project.tar.gz"

            git_absolute_dir = strip(read(`git rev-parse --show-toplevel`, String))
            cd(git_absolute_dir) do
                files = readlines(`git ls-files -c -m -o --exclude-standard`)
                unique!(files)
                existing_files = filter(file -> isfile(file) && file != project_archive, files)
                run(`tar -czf $project_archive $(existing_files)`)
            end

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
                    _addprocs_one_host(host, 1)
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
                _addprocs_one_host(host)
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

function _addprocs_one_host(host::Dict, workers_n=-1) # if workers_n = -1, it will take workers_n from host
    Distributed.addprocs(
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
end

get_unique_machine_ids() = unique(id -> Distributed.get_bind_addr(id), procs())
get_unique_machine_ips() = unique(map(id -> Distributed.get_bind_addr(id), procs()))


"""
    cluster_status!(cluster)

Run a status check on each machine in the list and throws an error if any of them is unreachable.
"""
function cluster_status!(cluster::Vector{<:Dict})
    printstyled("\n\n\nChecking machine main:\n", bold=true, color=:magenta)
    output = read(`julia -v`, String)
    println("Main host julia version: "*output)

    connection_error = []
    for node in cluster
        printstyled("Checking machine $(node[:host_address]):\n", bold=true, color=:magenta)
        try
            output = read(`ssh -i $(node[:private_key_path]) $(node[:host_address]) julia -v`, String)
            println("Host available, julia version: "*output)
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

    printstyled("\nAll machines are reachable\n\n", bold=true, color=:green)
end

end
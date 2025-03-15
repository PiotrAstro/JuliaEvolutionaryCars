# check https://github.com/albheim/DistributedEnvironments.jl/tree/main for original implementation
module DistributedEnvironments

import Pkg
import Logging
using Distributed, MacroTools

export @eachmachine, @eachworker, @initcluster, cluster_status!, remove_workers!, addprocs_one_host!, Cluster, @cluster_foreach!

struct Cluster
    main::Dict
    hosts::Vector
    initialization_function
    free_pids::Channel{Int}
    working_number::Threads.Atomic{Int}
end

function Cluster(main_host::Dict, remote_hosts::Vector, initialization_function)
    sum_of_workers = (length(remote_hosts) > 0 ? sum(host[:use_n_workers] for host in remote_hosts) : 0) + main_host[:use_n_workers]
    free_pids = Channel{Int}(sum_of_workers)
    working_number = Threads.Atomic{Int}(0)
    Cluster(main_host, remote_hosts, initialization_function, free_pids, working_number)
end

macro initcluster(cluster, tmp_dir_name, copy_env_and_code)
    return _initcluster(cluster, tmp_dir_name, copy_env_and_code)
end

"""
Function to add workers to the cluster.
It will automatically set up workers.
It will copy code and project info and precompile them, if copy_env_and_code is set to true.
Otherwise it will not do it, it will just go to the right place and activate env, without instantiiate it, it should only be used if one already copied everything and it is all set up.
"""
function _initcluster(
    cluster,
    tmp_dir_name_macro_input,
    copy_env_and_code_main_macro_input,
)
    quote
        cluster_tmp = $(esc(cluster))
        copy_env_and_code = $(esc(copy_env_and_code_main_macro_input))
        tmp_dir_name = $(esc(tmp_dir_name_macro_input))
        cluster_config_main = cluster_tmp.main

        remove_workers!()

        cluster_status!(cluster_tmp.hosts)
        for host in cluster_tmp.hosts
            host[:dir_project] = joinpath(host[:dir], tmp_dir_name)
        end

        if copy_env_and_code
            println("Copying project to all machines")
            project_archive = "_tmp_julia_comp_$(tmp_dir_name).tar.gz"
            create_project_archive(project_archive)
            try
                for host in cluster_tmp.hosts
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
            main_pids = Distributed.addprocs(
                cluster_config_main[:use_n_workers],
                env=["JULIA_BLAS_THREADS" => "$(cluster_config_main[:blas_threads_per_worker])"],
                exeflags="--threads=$(cluster_config_main[:julia_threads_per_worker])",
                enable_threaded_blas=cluster_config_main[:blas_threads_per_worker] > 1,
                topology=:master_worker
            )
            println("Added main host\n")
            for pid in main_pids
                put!(cluster_tmp.free_pids, pid)
            end

            for host in cluster_tmp.hosts
                printstyled("Adding $(host[:host_address]) -> $(host[:use_n_workers]) workers\n", bold=true, color=:magenta)
                host[:pids] = addprocs_one_host!(host)
                println("Added $(host[:host_address])\n")
                for pid in host[:pids]
                    put!(cluster_tmp.free_pids, pid)
                end
            end

            printstyled("All workers added\n", bold=true, color=:green)
        catch e
            println("Failed adding some workers, you should check it manually and change / remove these hosts")
            throw(e)
        end

        all_pids = Distributed.procs()
        cluster_tmp.initialization_function(all_pids)
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
    return added_pids
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
function hosts_manager(cluster::Cluster, should_exit::Ref{Bool}, sleep_seconds=300)
    sleep_start_time = time()
    while !should_exit[]
        if time() - sleep_start_time < sleep_seconds
            sleep(2)
        else
            Logging.@info("Checking workers")
            workers_official_pool = Distributed.workers()
            for host in cluster.hosts
                Logging.@info("Checking workers on host $(host[:host_address])")
                workers_in_pool = [pid for pid in host[:pids] if pid in workers_official_pool]

                # Currently this check doesnt work, cause main task blocks this task so I do not get any result from ay of them
                # I just hope that when some workers exit I will get some information about it in cluster_foreach! macro
                # So that these workers will be removed and I will add new ones here

                # workers_unreachability = map(workers_in_pool) do pid
                #     @async begin
                #         channel = Channel{Bool}(1)
                #         @async begin
                #             sleep(check_timeout)
                #             put!(channel, false)
                #         end
                #         @async begin
                #             try
                #                 # Simple health check
                #                 remotecall_fetch(() -> 1, pid)
                #                 put!(channel, true)
                #             catch e
                #                 if isa(e, TaskFailedException)
                #                     put!(channel, false)
                #                 elseif isa(e, ProcessExitedException)
                #                     put!(channel, false)
                #                 else
                #                     throw(e)
                #                 end
                #             end
                #         end
                #         return take!(channel)
                #     end
                # end
                # Logging.@info("Checks on $(host[:host_address]) scheduled")
                # workers_unreachability = fetch.(workers_unreachability)
                # display(workers_unreachability)
                # try
                #     Distributed.rmprocs(workers_in_pool[workers_unreachability]; waitfor=sleep_seconds)
                # catch
                #     Logging.@warn("Failed to remove dead workers")
                # end
                # host[:pids] = workers_in_pool[.!workers_unreachability]
                host[:pids] = workers_in_pool
                if length(host[:pids]) < host[:use_n_workers]
                    Logging.@warn("Some workers died on host $(host[:host_address])")
                    result, e = host_status(host)
                    if result
                        try
                            pids_n = host[:use_n_workers] - length(host[:pids])
                            new_pids = addprocs_one_host!(host, pids_n)
                            host[:pids] = vcat(host[:pids], new_pids)
                            cluster.initialization_function(new_pids)
                            for pid in new_pids
                                put!(cluster.free_pids, pid)
                            end
                            Logging.@info("Added new workers $new_pids to $(host[:host_address])")
                        catch
                            Logging.@warn("Failed to add new workers to $(host[:host_address])")
                        end
                    else !result
                        Logging.@warn(e)
                    end
                end
            end
            Logging.@info("Checking workers finished, waiting $sleep_seconds seconds till next check")
            sleep_start_time = time()
        end
    end

    Logging.@info("Exiting host manager")
end

macro cluster_foreach!(f, cluster, iterable, progress_meter_func=nothing, constants=[])
    # Notice we return an escaped expression
    return quote
        # Access everything through the properly escaped variables
        local _cluster = $(esc(cluster))
        local _f = $(esc(f))
        local _iterable = $(esc(iterable))
        local _progress_meter_func = $(esc(progress_meter_func))
        local _constants = $(esc(constants))
        
        local should_exit = Ref(false)
        local host_manager_task = Threads.@spawn hosts_manager(_cluster, should_exit)
        local items_array = reverse(_iterable)
        local array_locker = Threads.ReentrantLock()
        local progress_meter_lock = Threads.ReentrantLock()

        try
            while true
                if length(items_array) > 0
                    local pid = take!(_cluster.free_pids)
                    local item = lock(array_locker) do
                        pop!(items_array)
                    end
                    @async begin
                        pid_local = pid
                        Threads.atomic_add!(_cluster.working_number, 1)
                        try
                            # Now we're using the escaped function
                            fetch(Distributed.@spawnat pid_local _f(item, _constants...))
                            put!(_cluster.free_pids, pid_local)
                            lock(progress_meter_lock) do
                                if !isnothing(_progress_meter_func)
                                    _progress_meter_func()
                                end
                            end
                        catch e
                            if isa(e, ProcessExitedException) || isa(e, IOError)
                                Logging.@warn("Worker $(pid_local) exited with code $(e.exitcode)")
                                # check if it is in workers, remove if so
                                if pid_local in Distributed.workers()
                                    Logging.@warn("Removing worker $(pid_local)")
                                    Distributed.rmprocs(pid_local)
                                end
                            elseif isa(e, RemoteException)
                                Logging.@warn("Worker $(pid_local) exited with error: $(e)")
                                put!(_cluster.free_pids, pid_local)
                            else 
                                throw(e)
                            end
                            Logging.@warn("Some errors occured, will rerun $(item)")
                            lock(array_locker) do
                                push!(items_array, item)
                            end
                        end
                        Threads.atomic_sub!(_cluster.working_number, 1)
                    end
                elseif _cluster.working_number[] == 0
                    break
                else
                    sleep(2)
                end
            end
        finally
            should_exit[] = true
            wait(host_manager_task)
        end
    end
end

end
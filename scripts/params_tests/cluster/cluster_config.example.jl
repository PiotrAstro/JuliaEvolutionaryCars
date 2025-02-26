# -----------------------------------------------------------------------
# COPY THIS CONTENT TO cluster_config.jl
# -----------------------------------------------------------------------



# this file is used to store information about different cluster machines
# it should be gitinored - it has some sensitive information, like password etc

# these are used only in this script to put correct data to dicts
tmp_julia_threads_per_worker = 1
tmp_blas_threads_per_worker = 1  # actually I think it should always be 1, it is better to paralellise in higher level, using julia threads

# -----------------------------------------------------------------------
# REAL CONFIG
# -----------------------------------------------------------------------

CLUSTER_CONFIG_MAIN = Dict(
    :use_n_workers => 0,  # if set to 0, main will only collect data from other workers, should be int
    :blas_threads_per_worker => tmp_blas_threads_per_worker,
    :julia_threads_per_worker => tmp_julia_threads_per_worker,
)

RESULT_CHANNEL_BUFFER_SIZE = 16  # how many results main worker can hold, if it exceeds, remote workers will wait until main worker will take some results
CHECK_HOSTS_EACH_N_SECONDS = 60 * 3  # how often to check if hosts are still alive and add new ones if needed
CHECK_WORKER_TIMEOUT = 20  # how long to wait for worker to respond

# cluster settings:
private_key_path = raw"C:\Users\username\.ssh\private_key"

# It will copy code and project info and precompile them, if copy_env_and_code is set to true.
# Otherwise it will not do it, it will just go to the right place and activate env, without instantiiate it, it should only be used if one already copied everything and it is all set up.
# this copying uses git on main machine, other machines do not need git
COPY_ENV_AND_CODE = true
TMP_DIR_NAME = "_tmp_julia_comp_JuliaEvolutionaryCars"  # theoretically I do not have to change it, but in practice sometimes these folders are locked, so it is better to do so

CLUSTER_CONFIG_HOSTS = [
    Dict(
        # changable stuff
        :host_address => "User@10.10.10.10",
        :use_n_workers => 23,  # int
        :dir => raw"Documents",  # I think for scp it should rather be relative path

        # mostly same for all hosts
        :private_key_path => private_key_path,
        :blas_threads_per_worker => tmp_blas_threads_per_worker,
        :julia_threads_per_worker => tmp_julia_threads_per_worker,

        # mostly constant stuff
        :tunnel => false,
        :shell => :wincmd,
    ),
]

# shell=:posix: a POSIX-compatible Unix/Linux shell (sh, ksh, bash, dash, zsh, etc.). The default.
# shell=:csh: a Unix C shell (csh, tcsh).
# shell=:wincmd: Microsoft Windows cmd.exe.

# Notes and settings:
# -----------------------------------------------------------------------
# Julia version:
# Julia version should be the same on every machine
# I should have a separate channel and set my julia to it (not release, cause it will print strange statements to update)
# to do so:
# 1. install juliaup - msstore (preferable) or installer at https://github.com/JuliaLang/juliaup
# 2. "juliaup status" to see currently installed channels
# 3. "juliaup add 1.11.2" to add new channel if it doest exist, important! there should exist channel with this name, not only this version in release channel
# 4. "juliaup default 1.11.2" to set this channel as default
# 5. "juliaup status" to see if it is set correctly

# In the future, if I would like to specify which julia version to use for different runs, I could add exename entry and set it 
# to use some channel, e.g. "exename" => "julia +1.11.2"

# -----------------------------------------------------------------------
# General info:
# I have scripts that will copy all files tracked by git from main machine to all other machines
# So make sure that git is working correctly on main machine
# it runs:

# files = readlines(`git ls-files -c -m -o --exclude-standard`)
# unique!(files)
# existing_files = filter(file -> isfile(file) && file != project_archive, files)

# so it will list files:
# -c: shows cached/tracked files
# -m: shows modified tracked files
# -o: shows other (untracked) files
# --exclude-standard: only excludes files that are in .gitignore
# then remove duplicates
# and filter those that were recently deleted

# Project.toml shoud be in the main git folder
# So eventually if the current project has git, it will get all files in the current form (not one from git, but local one) and exclude those from gitignore

# -----------------------------------------------------------------------
# SSH keys:
# Generating keys for ssh, on client (your_file_name is private, your_file_name.pub is public):
# ssh-keygen -t rsa -b 4096 -m PEM -f C:\Users\YourUsername\.ssh\your_file_name

# Now on server I can use same key on all servers, I should do on each server:
# open or create file C:\Users\username\.ssh\authorized_keys  if username is not in administrator group
# open or create file C:\ProgramData\ssh\administrators_authorized_keys if username is in administrator group (it doesnt have to be admin, but to have admin rights)
# paste to this file content of your_file_name.pub

# -----------------------------------------------------------------------
# SSH server
# I should check for ssh server in settings -> system -> additional functions and add ssh server
# Then go to "Usługi" and start ssh server

# then add these rules in admin powershell to let ssh through firewall:
# New-NetFirewallRule -Name "SSH" -DisplayName "SSH" -Description "Allow SSH" -Direction Inbound -Protocol TCP -LocalPort 22 -Action Allow
# New-NetFirewallRule -Name "SSH-Out" -DisplayName "SSH" -Description "Allow SSH" -Direction Outbound -Protocol TCP -LocalPort 22 -Action Allow

# also I may have to add Julia to allowed apps through firewall (Julia establish its own connection separate from ssh)
# I should go to "Panel sterowania\System i zabezpieczenia\Zapora Windows Defender\Dozwolone aplikacje"
# then add real path to julia, e.g. C:\Users\admin\.julia\juliaup\julia-1.11.2+0.x64.w64.mingw32\bin\julia.exe

# -----------------------------------------------------------------------
# MKL
# for MKL on windows I should install smth like:
# https://aka.ms/vs/17/release/vc_redist.x64.exe
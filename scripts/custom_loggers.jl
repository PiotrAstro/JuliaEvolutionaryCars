module CustomLoggers

export PlainInfoLogger, SimpleFileLogger

using Logging
import Distributed
import LoggingExtras
import Dates


# ---------------------------------------------------------------------------------------------------------------------------------
abstract type MyLoggers <: Logging.AbstractLogger end

function Logging.min_enabled_level(logger::MyLoggers)
    return logger.min_level
end

# Determine if the logger should handle a message based on log level
Logging.shouldlog(logger::MyLoggers, level::LogLevel, _module, group, id) = level >= logger.min_level

# ---------------------------------------------------------------------------------------------------------------------------------

struct SimpleFileLogger <: MyLoggers
    min_level::LogLevel
    wrapped_logger::LoggingExtras.FileLogger
    print_logged::Bool
    locker::ReentrantLock
end

SimpleFileLogger(file_name::String) = SimpleFileLogger(file_name, false)
SimpleFileLogger(file_name::String, print_logged::Bool) = SimpleFileLogger(file_name, Info, print_logged)
SimpleFileLogger(file_name::String, min_level::LogLevel, print_logged::Bool) = SimpleFileLogger(min_level, LoggingExtras.FileLogger(file_name), print_logged, ReentrantLock())

function Logging.handle_message(logger::SimpleFileLogger, level::LogLevel, message, _module, group, id, filepath, line; kwargs...)
    lock(logger.locker)
    try
        Logging.handle_message(logger.wrapped_logger, level, message, _module, group, id, filepath, line; kwargs...)
    finally
        unlock(logger.locker)
    end

    if logger.print_logged
        logged_text = string("\n", level, ": ", Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS"), "\n", message)
        println(logged_text)
    end
end

# ---------------------------------------------------------------------------------------------------------------------------------
# Define a custom logger type
struct PlainInfoLogger <: MyLoggers
    min_level::LogLevel
    locker::ReentrantLock
    stream::IO
end

PlainInfoLogger() = PlainInfoLogger(Info, ReentrantLock(), stdout)
PlainInfoLogger(min_level::LogLevel) = PlainInfoLogger(min_level, ReentrantLock(), stdout)

# Define how to handle logging messages for different levels
function Logging.handle_message(logger::PlainInfoLogger, level::LogLevel, message, _module, group, id, filepath, line; kwargs...)
    lock(logger.locker)
        if level == Info
            print(logger.stream, message)  # Print message directly for Info level
        else
            # Add formatting for other log levels
            prefix = string(level, ": ")
            print(logger.stream, prefix, message)
        end
    unlock(logger.locker)
end

# ---------------------------------------------------------------------------------------------------------------------------------

struct RemoteLogger <: MyLoggers
    min_level::LogLevel
    locker::ReentrantLock
    main_process_id::Int
end

RemoteLogger() = RemoteLogger(Info, ReentrantLock(),  1)
RemoteLogger(main_process_id::Int) = RemoteLogger(Info, ReentrantLock(), main_process_id)

function call_on_main_process(level, message)
    Logging.@logmsg(level, message)
end

function Logging.handle_message(logger::RemoteLogger, level::LogLevel, message, _module, group, id, filepath, line; kwargs...)
    lock(logger.locker)
    try
        Distributed.remotecall(CustomLoggers.call_on_main_process, logger.main_process_id, level, message)
    finally
        unlock(logger.locker)
    end
end

end

module CustomLoggers

export PlainInfoLogger

using Logging

# Define a custom logger type
struct PlainInfoLogger <: AbstractLogger
    min_level::LogLevel
    stream::IO
end

PlainInfoLogger() = PlainInfoLogger(Info, stdout)

# Define how to handle logging messages for different levels
function Logging.handle_message(logger::PlainInfoLogger, level::LogLevel, message, _module, group, id, filepath, line; kwargs...)
    if level == Info
        print(logger.stream, message)  # Print message directly for Info level
    else
        # Add formatting for other log levels
        prefix = string(level, ": ")
        print(logger.stream, prefix, message)
    end
end

function Logging.min_enabled_level(logger::PlainInfoLogger)
    return logger.min_level
end

# Determine if the logger should handle a message based on log level
Logging.shouldlog(logger::PlainInfoLogger, level::LogLevel, _module, group, id) = level >= logger.min_level


# ---------------------------------------------------------------------------------------------------------------------------------

end
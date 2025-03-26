module Tokenization

export BPETokenizer, construct, save, load, encode, decode, load_and_encode

using Mmap

"""
    ConstructWord

Helper struct for tokenizer construction process
"""
struct ConstructWord
    word::String
    tokens::Vector{Int}
    pairs::Vector{Tuple{Int, Int}}
    counter::Int
end

Base.hash(cw::ConstructWord) = hash(cw.word)
Base.:(==)(cw1::ConstructWord, cw2::ConstructWord) = cw1.word == cw2.word

"""
    BPETokenizer

Byte Pair Encoding tokenizer implementation
"""
mutable struct BPETokenizer
    encoding::String
    special_tokens::Vector{String}
    special_tokens_reverse::Dict{String, Int}
    special_tokens_split_pattern::Regex
    bytes_to_unicode_list::Vector{String}
    unicode_to_bytes::Dict{String, Int}
    split_formula::Regex
    tokens_mapping::Vector{Vector{UInt8}}
    tokens_merge_ranking::Dict{Tuple{Int, Int}, Int}
    cache::Dict{String, Vector{Int}}

    function BPETokenizer(special_tokens::Union{Nothing, Vector{String}}=nothing)
        # Constructor with minimal initialization
        special_tokens = isnothing(special_tokens) ? construct_special_tokens() : special_tokens
        special_tokens_reverse = Dict(token => position for (position, token) in enumerate(special_tokens))
        special_tokens_pattern = Regex(join(map(escape_string, special_tokens), "|"))
        
        bytes_to_unicode_list = bytes_to_unicode()
        unicode_to_bytes = Dict(char => i-1 for (i, char) in enumerate(bytes_to_unicode_list))
        
        # Pattern to split text into words
        split_formula = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        
        new(
            "utf-8",
            special_tokens,
            special_tokens_reverse,
            special_tokens_pattern,
            bytes_to_unicode_list,
            unicode_to_bytes,
            split_formula,
            Vector{Vector{UInt8}}(),
            Dict{Tuple{Int, Int}, Int}(),
            Dict{String, Vector{Int}}()
        )
    end
end

"""
    decode(tokenizer::BPETokenizer, tokens::Vector{Int}, error::String="replace")

Decode tokens back to text
"""
function decode(tokenizer::BPETokenizer, tokens::Vector{Int}, error::String="replace")
    # Check if tokenizer is initialized
    isempty(tokenizer.tokens_mapping) && error("Tokenizer not initialized or loaded")

    # Convert tokens to byte arrays
    byteseqs = Vector{UInt8}[]
    for token_id in tokens
        if token_id < length(tokenizer.tokens_mapping)
            push!(byteseqs, tokenizer.tokens_mapping[token_id+1])
        else
            # Handle special tokens
            special_token = tokenizer.special_tokens[token_id - length(tokenizer.tokens_mapping) + 1]
            push!(byteseqs, Vector{UInt8}(special_token))
        end
    end
    
    # Join byte arrays and decode to text
    final_bytes = reduce(vcat, byteseqs, init=UInt8[])
    
    # Handle errors in text conversion according to parameter
    if error == "strict"
        return String(final_bytes)
    else
        # Replace invalid sequences
        return replace_invalid_utf8(final_bytes)
    end
end

"""
    replace_invalid_utf8(bytes::Vector{UInt8})

Replace invalid UTF-8 sequences with replacement character
"""
function replace_invalid_utf8(bytes::Vector{UInt8})
    try
        return String(bytes)
    catch
        # Simple implementation - in a full version this would be more sophisticated
        return String(map(b -> isvalid(Char, b) ? b : 0xEF, bytes))
    end
end

"""
    load_and_encode(tokenizer::BPETokenizer, path::String, delimiter::String="<|endoftext|>")

Load text from file(s) and encode with initialized tokenizer
"""
function load_and_encode(tokenizer::BPETokenizer, path::String, delimiter::String="<|endoftext|>")
    # Check if tokenizer is initialized
    isempty(tokenizer.tokens_mapping) && error("Tokenizer not initialized or loaded")
    
    # Load texts
    texts = load_texts(tokenizer, path)
    
    # Encode all texts with delimiter
    encoded = Int[]
    for text in texts
        append!(encoded, encode(tokenizer, text))
        append!(encoded, encode(tokenizer, delimiter))
    end
    
    return encoded
end

"""
    encode(tokenizer::BPETokenizer, text::String)

Encode text to tokens
"""
function encode(tokenizer::BPETokenizer, text::String)
    # Check if tokenizer is initialized
    isempty(tokenizer.tokens_mapping) && error("Tokenizer not initialized or loaded")
    
    final_encoded = Int[]
    
    # Split by special tokens pattern
    parts = split(text, tokenizer.special_tokens_split_pattern, keepempty=true)
    
    for (i, text_part) in enumerate(parts)
        # Check if this part is a special token
        is_special = i % 2 == 0
        
        if is_special
            token_id = tokenizer.special_tokens_reverse[text_part] + length(tokenizer.tokens_mapping)
            push!(final_encoded, token_id)
        else
            # Find all words according to the split formula
            words = [m.match for m in eachmatch(tokenizer.split_formula, text_part)]
            
            for word in words
                # Convert each word to tokens
                word_tokens = word_to_tokens(tokenizer, word)
                append!(final_encoded, word_tokens)
            end
        end
    end
    
    return final_encoded
end

"""
    is_special_token(tokenizer::BPETokenizer, token::Int, special_token::String)

Check if a token corresponds to a specific special token
"""
function is_special_token(tokenizer::BPETokenizer, token::Int, special_token::String)
    if token >= length(tokenizer.tokens_mapping)
        return tokenizer.special_tokens[token - length(tokenizer.tokens_mapping) + 1] == special_token
    end
    return false
end

"""
    get_tokens_number(tokenizer::BPETokenizer)

Get total number of tokens in vocabulary
"""
function get_tokens_number(tokenizer::BPETokenizer)
    return length(tokenizer.tokens_mapping) + length(tokenizer.special_tokens)
end

"""
    construct(tokenizer::BPETokenizer, path::String, token_normal_number::Int)

Construct a tokenizer from a corpus of text
"""
function construct(tokenizer::BPETokenizer, path::String, token_normal_number::Int)
    words = Dict{String, ConstructWord}()
    
    # Load texts
    texts = load_texts(tokenizer, path)
    
    for text_part in texts
        # Split by special tokens
        parts = split(text_part, tokenizer.special_tokens_split_pattern, keepempty=true)
        
        for (i, text_part_split) in enumerate(parts)
            is_special = i % 2 == 0
            
            if !is_special
                # Find all words according to the split formula
                for word in [m.match for m in eachmatch(tokenizer.split_formula, text_part_split)]
                    if haskey(words, word)
                        words[word] = ConstructWord(
                            words[word].word,
                            words[word].tokens,
                            words[word].pairs,
                            words[word].counter + 1
                        )
                    else
                        # Convert to bytes
                        word_tokens = [b for b in Vector{UInt8}(word)]
                        
                        if length(word_tokens) > 1
                            # Create pairs
                            pairs = [(word_tokens[i], word_tokens[i+1]) for i in 1:length(word_tokens)-1]
                            
                            words[word] = ConstructWord(
                                word,
                                word_tokens,
                                pairs,
                                1
                            )
                        end
                    end
                end
            end
        end
    end
    
    # Construct vocabulary from word tokens
    construct_word_tokens(tokenizer, collect(values(words)), token_normal_number)
end

"""
    save(tokenizer::BPETokenizer, path::String)

Save tokenizer vocabulary to a file
"""
function save(tokenizer::BPETokenizer, path::String)
    # Check if tokenizer is initialized
    isempty(tokenizer.tokens_mapping) && error("Tokenizer not initialized or loaded")
    
    # Create reversed ranking dictionary
    reversed_ranking = Dict(position => merge for (merge, position) in tokenizer.tokens_merge_ranking)
    
    # Ensure directory exists
    mkpath(dirname(path))
    
    open(path, "w") do f
        for (position, token_map) in enumerate(tokenizer.tokens_mapping)
            if haskey(reversed_ranking, position-1)  # Julia is 1-indexed
                pos1, pos2 = reversed_ranking[position-1]
                println(f, "$(bytes_to_unicode(tokenizer, token_map)) $pos1 $pos2")
            else
                println(f, bytes_to_unicode(tokenizer, token_map))
            end
        end
    end
end

"""
    load(tokenizer::BPETokenizer, path::String)

Load a tokenizer vocabulary from a file
"""
function load(tokenizer::BPETokenizer, path::String)
    tokenizer.tokens_mapping = Vector{UInt8}[]
    tokenizer.tokens_merge_ranking = Dict{Tuple{Int, Int}, Int}()
    
    open(path, "r") do f
        for line in readlines(f)
            parts = split(line)
            
            if length(parts) == 1
                # Just a token
                push!(tokenizer.tokens_mapping, unicode_to_bytes(tokenizer, parts[1]))
            elseif length(parts) == 3
                # Token with its pair origins
                token_text, first_pair_text, second_pair_text = parts
                push!(tokenizer.tokens_mapping, unicode_to_bytes(tokenizer, token_text))
                
                key = (parse(Int, first_pair_text), parse(Int, second_pair_text))
                tokenizer.tokens_merge_ranking[key] = length(tokenizer.tokens_mapping) - 1
            else
                error("Error loading token: invalid format")
            end
        end
    end
end

"""
    load_texts(tokenizer::BPETokenizer, path::String)

Load text corpus from file(s)
"""
function load_texts(tokenizer::BPETokenizer, path::String)
    documents_to_read = String[]
    
    if isdir(path)
        # Read all files in directory
        for file in readdir(path)
            push!(documents_to_read, joinpath(path, file))
        end
    else
        # Read single file
        push!(documents_to_read, path)
    end
    
    text_parts = String[]
    
    for file in documents_to_read
        open(file, "r") do f
            push!(text_parts, read(f, String))
        end
    end
    
    return text_parts
end

"""
    unicode_to_bytes(tokenizer::BPETokenizer, unicode_word::String)

Convert a Unicode string to bytes using the tokenizer's mapping
"""
function unicode_to_bytes(tokenizer::BPETokenizer, unicode_word::String)
    return [tokenizer.unicode_to_bytes[char] for char in unicode_word]
end

"""
    bytes_to_unicode(tokenizer::BPETokenizer, bytes_word::Vector{UInt8})

Convert bytes to a Unicode string using the tokenizer's mapping
"""
function bytes_to_unicode(tokenizer::BPETokenizer, bytes_word::Vector{UInt8})
    return join([tokenizer.bytes_to_unicode_list[byte+1] for byte in bytes_word])
end

"""
    word_to_tokens(tokenizer::BPETokenizer, word::String)

Convert a word to tokens using the learned merges
"""
function word_to_tokens(tokenizer::BPETokenizer, word::String)
    # Check cache
    if haskey(tokenizer.cache, word)
        return tokenizer.cache[word]
    end
    
    not_valid_token = length(tokenizer.tokens_mapping)
    word_bytes = Vector{UInt8}(word)
    word_tokens = Int[b for b in word_bytes]
    
    pairs = []
    for i in 1:length(word_tokens)-1
        pair = (word_tokens[i], word_tokens[i+1])
        ranking = get(tokenizer.tokens_merge_ranking, pair, not_valid_token)
        push!(pairs, (pair, ranking))
    end
    
    while !isempty(pairs)
        # Find pair with lowest ranking (highest priority)
        argmin_idx = argmin([p[2] for p in pairs])
        new_token = pairs[argmin_idx][2]
        
        if new_token == not_valid_token
            break
        else
            # Apply merge
            deleteat!(word_tokens, argmin_idx)
            word_tokens[argmin_idx] = new_token
            
            # Update pairs
            deleteat!(pairs, argmin_idx)
            
            # Update affected pairs
            argmin_prev = argmin_idx - 1
            argmin_next = argmin_idx
            
            if argmin_next < length(pairs)
                pair = (word_tokens[argmin_next], word_tokens[argmin_next + 1])
                ranking = get(tokenizer.tokens_merge_ranking, pair, not_valid_token)
                pairs[argmin_next] = (pair, ranking)
            end
            
            if argmin_prev >= 1
                pair = (word_tokens[argmin_prev], word_tokens[argmin_prev + 1])
                ranking = get(tokenizer.tokens_merge_ranking, pair, not_valid_token)
                pairs[argmin_prev] = (pair, ranking)
            end
        end
    end
    
    # Cache result
    tokenizer.cache[word] = word_tokens
    return word_tokens
end

"""
    construct_word_tokens(tokenizer::BPETokenizer, word_list::Vector{ConstructWord}, desired_tokens_n::Int)

Build the tokenizer's vocabulary from a list of words with their frequencies
"""
function construct_word_tokens(tokenizer::BPETokenizer, word_list::Vector{ConstructWord}, desired_tokens_n::Int)
    # Initialize with byte tokens
    tokens_mapping = [UInt8[i-1] for i in 1:256]
    pairs_counter = Dict{Tuple{Int, Int}, Int}()
    pairs_word_backbond = Dict{Tuple{Int, Int}, Set{ConstructWord}}()
    tokens_merge_ranking = Dict{Tuple{Int, Int}, Int}()
    
    # Count initial pairs
    for word in word_list
        for pair in word.pairs
            pairs_counter[pair] = get(pairs_counter, pair, 0) + word.counter
            word_set = get(pairs_word_backbond, pair, Set{ConstructWord}())
            push!(word_set, word)
            pairs_word_backbond[pair] = word_set
        end
    end
    
    # Iteratively merge the most frequent pairs
    while length(tokens_mapping) < desired_tokens_n
        print_progress(length(tokens_mapping), desired_tokens_n)
        
        # Find most frequent pair
        max_pair = first(keys(pairs_counter))
        max_count = pairs_counter[max_pair]
        
        for (pair, count) in pairs_counter
            if count > max_count
                max_pair = pair
                max_count = count
            end
        end
        
        # Create new token
        new_token_value = length(tokens_mapping)
        tokens_merge_ranking[max_pair] = new_token_value
        push!(tokens_mapping, vcat(tokens_mapping[max_pair[1]+1], tokens_mapping[max_pair[2]+1]))
        
        # Update pairs in affected words
        for word in pairs_word_backbond[max_pair]
            word_tokens = word.tokens
            pairs = word.pairs
            
            # Find and apply all instances of the pair
            i = 1
            while i <= length(pairs)
                if pairs[i] == max_pair
                    # Apply merge
                    deleteat!(word_tokens, i)
                    word_tokens[i] = new_token_value
                    
                    # Remove the merged pair
                    deleteat!(pairs, i)
                    
                    # Update affected pairs
                    for pos in [i-1, i]
                        if 1 <= pos <= length(pairs)
                            # Create new pair
                            pair = (word_tokens[pos], word_tokens[pos+1])
                            pairs[pos] = pair
                            
                            # Update pair statistics
                            pairs_counter[pair] = get(pairs_counter, pair, 0) + word.counter
                            word_set = get(pairs_word_backbond, pair, Set{ConstructWord}())
                            push!(word_set, word)
                            pairs_word_backbond[pair] = word_set
                        end
                    end
                else
                    i += 1
                end
            end
        end
        
        # Remove merged pair from counters
        delete!(pairs_counter, max_pair)
        delete!(pairs_word_backbond, max_pair)
    end
    
    # Store final vocabulary
    tokenizer.tokens_mapping = tokens_mapping
    tokenizer.tokens_merge_ranking = tokens_merge_ranking
    
    # Clear progress
    print_progress(desired_tokens_n, desired_tokens_n)
    println()
end

"""
    print_progress(current::Int, total::Int)

Print a progress indicator for token construction
"""
function print_progress(current::Int, total::Int)
    print("\r", " "^50, "\r")  # Clear line
    print("Constructing tokens: $current / $total")
end

"""
    bytes_to_unicode()

Create a reversible mapping between bytes and printable unicode characters
"""
function bytes_to_unicode()
    # Initial ASCII printable characters
    ascii_printable = vcat(
        collect(Int('!'):Int('~')),
        collect(Int('¡'):Int('¬')),
        collect(Int('®'):Int('ÿ'))
    )
    
    byte_size = 256
    above_ascii = 0
    bytes_printable = String[]
    
    for byte in 0:byte_size-1
        if byte in ascii_printable
            push!(bytes_printable, string(Char(byte)))
        else
            push!(bytes_printable, string(Char(byte_size + above_ascii)))
            above_ascii += 1
        end
    end
    
    return bytes_printable
end

"""
    construct_special_tokens()

Define special tokens for the tokenizer
"""
function construct_special_tokens()
    return ["<|endoftext|>"]
end

end # module Tokenization

# Script to build and save tokenizer
import .Tokenization

# Constants
const TOKENS_N = 32_768
const DATA_PATH = joinpath(raw"C:\Piotr\AIProjects\MyLLM\data\fineweb\train")
const SAVE_PATH = joinpath(raw"C:\Piotr\AIProjects\MyLLM\results", "tokenizer_julia_fineweb_whole_$(TOKENS_N).txt")

# Main execution block
begin
    println("Building tokenizer with $(TOKENS_N) tokens")
    
    # Create tokenizer
    tokenizer = Tokenization.BPETokenizer()
    
    # Construct tokenizer from data
    Tokenization.construct(tokenizer, DATA_PATH, TOKENS_N - 1)  # additional one will be for eof
    
    # Save tokenizer
    Tokenization.save(tokenizer, SAVE_PATH)
    
    println("Tokenizer successfully saved to: $SAVE_PATH")
end
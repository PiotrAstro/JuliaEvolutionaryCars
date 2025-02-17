
using InteractiveUtils

function check()
    array = ones(10)
    @code_llvm array[5]
end

check()
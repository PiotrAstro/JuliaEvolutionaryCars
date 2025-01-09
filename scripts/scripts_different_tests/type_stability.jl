module Stability

import Flux
using InteractiveUtils

struct ABC{T<:Flux.Chain}
    sn::T
end

function use1(abc::ABC)
    abc.sn(zeros(10))
end

abc = ABC(Flux.Chain(x->x))



function test()
    # isconcretetype(ABC{Int})
    @code_warntype use1(abc)
end

end

import .Stability

Stability.test()
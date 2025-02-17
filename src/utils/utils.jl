module Utils

export GenieClust, PAM, FCM, StatesGrouping

include("Clustering/GenieClust.jl")
import .GenieClust

include("Clustering/PAM.jl")
import .PAM

include("Clustering/FCM.jl")
import .FCM

include("StatesGrouping/StatesGrouping.jl")
import .StatesGrouping

end
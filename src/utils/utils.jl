module Utils

export GenieClust, PAM, FCM, StatesGrouping

include("Clustering/GenieClust.jl")
import .GenieClust

include("Clustering/PAM.jl")
import .PAM

include("Clustering/FCM.jl")
import .FCM

include("Clustering/BanditPAM.jl")
import .BanditPAM

include("StatesGrouping/StatesGrouping.jl")
import .StatesGrouping

end
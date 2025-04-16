# ------------------------------------------------------------------------------------------
# getting distance membership levels
# ------------------------------------------------------------------------------------------
function distance_membership_levels(
    encoded_all_states::Matrix{F},
    encoded_exemplars::Matrix{F}; distance_metric::Symbol=:cosine, # :cosine or :euclidean or :cityblock or :mul
    method::Symbol=:hclust_complete, # :genie or :pam or :kmedoids
    mval::Int=2  # Fuzzy parameter, it is different that one in papers!
)::Vector{Vector{Vector{F}}} where {F<:AbstractFloat}
    if distance_metric == :cosine
        distance_premetric = Distances.CosineDist()
    elseif distance_metric == :euclidean
        distance_premetric = Distances.Euclidean()
    elseif distance_metric == :cityblock
        distance_premetric = Distances.Cityblock()
    elseif distance_metric == :mul
        distance_premetric = MulDistance()
    elseif distance_metric == :mul_weighted
        distance_premetric = MulWeightedDistance()
    elseif distance_metric == :mul_norm
        distance_premetric = MulNormDistance()
    else
        throw(ArgumentError("Unknown distance metric: $distance_metric"))
    end

    if method == :flat
        levels = flat_levels(encoded_exemplars)
    elseif method == :hclust_complete
        levels = hclust_levels(encoded_exemplars, distance_premetric, :complete)
    elseif method == :hclust_average
        levels = hclust_levels(encoded_exemplars, distance_premetric, :average)
    elseif method == :hclust_single
        levels = hclust_levels(encoded_exemplars, distance_premetric, :single)
    elseif method == :hclust_ward
        levels = hclust_levels(encoded_exemplars, distance_premetric, :ward)

    elseif method == :kmeans_exemplars_crisp
        levels = kmeans_levels(encoded_exemplars, encoded_exemplars, distance_premetric, mval, true)
    elseif method == :kmeans_all_crisp
        levels = kmeans_levels(encoded_all_states, encoded_exemplars, distance_premetric, mval, true)
    elseif method == :pam_exemplars_crisp
        levels = pam_levels(encoded_exemplars, encoded_exemplars, distance_premetric, mval, true)
    elseif method == :pam_all_crisp
        levels = pam_levels(encoded_all_states, encoded_exemplars, distance_premetric, mval, true)

    elseif method == :kmeans_exemplars_fuzzy
        levels = kmeans_levels(encoded_exemplars, encoded_exemplars, distance_premetric, mval, false)
    elseif method == :kmeans_all_fuzzy
        levels = kmeans_levels(encoded_all_states, encoded_exemplars, distance_premetric, mval, false)
    elseif method == :pam_exemplars_fuzzy
        levels = pam_levels(encoded_exemplars, encoded_exemplars, distance_premetric, mval, false)
    elseif method == :pam_all_fuzzy
        levels = pam_levels(encoded_all_states, encoded_exemplars, distance_premetric, mval, false)
    else
        throw(ArgumentError("Unknown method: $method"))
    end

    return levels
end

function flat_levels(encoded_exemplars::Matrix{F})::Vector{Vector{Vector{F}}} where {F<:AbstractFloat}
    exemplars_n = size(encoded_exemplars, 2)
    levels = [[zeros(F, exemplars_n) for _ in 1:exemplars_n]]
    for i in 1:exemplars_n
        levels[1][i][i] = one(F)
    end
    return levels
end

function hclust_levels(encoded_exemplars::Matrix{F}, distance_premetric, linkage_method::Symbol)::Vector{Vector{Vector{F}}} where {F<:AbstractFloat}
    states_n = size(encoded_exemplars, 2)
    distances = Distances.pairwise(distance_premetric, encoded_exemplars, encoded_exemplars)
    merges = Clustering.hclust(distances, linkage=linkage_method).merges
    root = _create_tree_hclust(merges)
    levels = Vector{Vector{Vector{F}}}()

    levels_nodes = Vector{Vector{TreeNode}}()
    push!(levels_nodes, [root.left, root.right])
    while true
        last_level = levels_nodes[end]
        new_level = Vector{TreeNode}()
        for node in last_level
            if !is_leaf(node)
                push!(new_level, node.left)
                push!(new_level, node.right)
            end
        end
        if isempty(new_level)
            break
        end
        push!(levels_nodes, new_level)
    end

    for level in levels_nodes
        level_distances = Vector{Vector{F}}()
        for node in level
            this_node_membership = zeros(F, states_n)
            this_node_membership[node.elements] .= one(F)
            push!(level_distances, this_node_membership)
        end
        push!(levels, level_distances)
    end

    return levels
end

function kmeans_levels(encoded_states_for_centroids::Matrix{F}, encoded_exemplars::Matrix{F}, distance_premetric, mval, crisp::Bool)::Vector{Vector{Vector{F}}} where {F<:AbstractFloat}
    exemplars_n = size(encoded_exemplars, 2)
    levels = Vector{Vector{Vector{F}}}()
    k = 2
    while k < exemplars_n
        centroids = Logging.with_logger(Logging.NullLogger()) do
            Clustering.kmeans(encoded_states_for_centroids, k; distance=distance_premetric).centers
        end
        membership = get_membership(encoded_exemplars, centroids, distance_premetric, mval, crisp=crisp)
        push!(levels, [
            Vector{F}(row) for row in eachrow(membership)
        ])
        k *= 2
    end

    tmp_last_level = [zeros(F, exemplars_n) for _ in 1:exemplars_n]
    for i in 1:exemplars_n
        tmp_last_level[i][i] = one(F)
    end
    push!(levels, tmp_last_level)

    return levels
end

function pam_levels(encoded_states_for_medoids::Matrix{F}, encoded_exemplars::Matrix{F}, distance_premetric, mval, crisp::Bool)::Vector{Vector{Vector{F}}} where {F<:AbstractFloat}
    exemplars_n = size(encoded_exemplars, 2)
    distances = Distances.pairwise(distance_premetric, encoded_states_for_medoids)
    levels = Vector{Vector{Vector{F}}}()
    k = 2
    while k < exemplars_n
        medoids_ids = PAM.faster_pam(distances, k)
        membership = get_membership(encoded_exemplars, encoded_states_for_medoids[:, medoids_ids], distance_premetric, mval, crisp=crisp)
        push!(levels, [
            Vector{F}(row) for row in eachrow(membership)
        ])
        k *= 2
    end

    tmp_last_level = [zeros(F, exemplars_n) for _ in 1:exemplars_n]
    for i in 1:exemplars_n
        tmp_last_level[i][i] = one(F)
    end
    push!(levels, tmp_last_level)

    return levels
end

function get_membership(from_points::Matrix{F}, to_points::Matrix{F}, distance::Symbol, mval; crisp::Bool=false)::Matrix{F} where {F<:AbstractFloat}
    if distance == :cosine
        distance_premetric = Distances.CosineDist()
    elseif distance == :euclidean
        distance_premetric = Distances.Euclidean()
    elseif distance == :cityblock
        distance_premetric = Distances.Cityblock()
    elseif distance_premetric == :mul
        distance_premetric = MulDistance()
    elseif distance_premetric == :mul_weighted
        distance_premetric = MulWeightedDistance()
    elseif distance_premetric == :mul_norm
        distance_premetric = MulNormDistance()
    else
        throw(ArgumentError("Unknown distance metric: $distance"))
    end
    return get_membership(from_points, to_points, distance_premetric, mval, crisp=crisp)
end

function get_membership(from_points::Matrix{F}, to_points::Matrix{F}, distance_premetric, mval; crisp::Bool=false)::Matrix{F} where {F<:AbstractFloat}
    distances = Distances.pairwise(distance_premetric, to_points, from_points)
    epsilon = F(1e-8)
    from_n = size(from_points, 2)
    to_n = size(to_points, 2)
    membership = zeros(F, to_n, from_n)

    for from_point_id in 1:from_n
        for to_point_id in 1:to_n
            @fastmath membership[to_point_id, from_point_id] = one(F) / (distances[to_point_id, from_point_id] + epsilon)^mval
        end
    end

    for memb_col in eachcol(membership)
        memb_col ./= sum(memb_col)
        if crisp
            max_ind = argmax(memb_col)
            memb_col .= zero(F)
            memb_col[max_ind] = one(F)
        end
    end

    return membership
end
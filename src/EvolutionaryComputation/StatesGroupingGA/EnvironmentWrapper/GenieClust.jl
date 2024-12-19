
# We can use either multithreaded or single threaded version
# comment or uncomment one line in _compute_mst_on_the_fly function
module GenieClust

export genie_clust

import DataStructures
import LinearAlgebra

"""
Performs genie clust on encoded states

currently supported distance_metrics:
:cosine
:euclidean

It outputs order of combining clusters
"""
function genie_clust(encoded_states::Matrix{Float32}; gini_threshold::Float64=0.3, distance_callable) :: Matrix{Int}
    edges = _compute_mst_on_the_fly(encoded_states, distance_callable)

    points_n = size(encoded_states, 2)
    result = Matrix{Int}(undef, points_n - 1, 2)
    
    clusters = Clusters(points_n)

    gini = 0.0
    # we will merge all except for last 2 clusters
    for i in 1:(points_n - 2)
        if gini < gini_threshold
            edge = pop!(edges)
            cl_ind1 = get_cluster_id(clusters, edge.ind1)
            cl_ind2 = get_cluster_id(clusters, edge.ind2)
        else
            cl_ind1, cl_ind2 = pop_min!(edges, clusters)
        end
        gini = merge_clusters!(clusters, cl_ind1, cl_ind2, gini, i)
        result[i, 1] = cl_ind1
        result[i, 2] = cl_ind2
    end
    # We will manualy merge last two clusters, so that gini will not be nan
    edge = pop!(edges)
    cl_ind1 = get_cluster_id(clusters, edge.ind1)
    cl_ind2 = get_cluster_id(clusters, edge.ind2)
    result[end, 1] = cl_ind1
    result[end, 2] = cl_ind2

    return result
end

# ------------------------------------------------------------------------------------------
# something like disjoint set, but their indices are increasing when union is performed
mutable struct MySet
    const parents::Vector{Int}
    current_fill::Int
end

MySet(n::Int) = MySet(fill(-1, n * 2 - 1), n)

function find_root(set::MySet, x::Int) :: Int
    if set.parents[x] == -1
        return x
    else
        set.parents[x] = p = find_root(set, set.parents[x])
        return p
    end
end

function union!(set::MySet, x::Int, y::Int) :: Int
    cf = set.current_fill = set.current_fill + 1
    x_root = find_root(set, x)
    y_root = find_root(set, y)
    @assert x_root != y_root
    set.parents[x_root] = cf
    set.parents[y_root] = cf
    return cf
end

# ------------------------------------------------------------------------------------------
# Edge declaration

struct Edge
    ind1::Int
    ind2::Int
    distance::Float32
end

Base.isless(e1::Edge, e2::Edge) = e1.distance < e2.distance

# ------------------------------------------------------------------------------------------
# methods with cluster sizes

mutable struct Clusters
    _set::MySet
    _min_cluster_count_id::Int
    _cluster_counts::Vector{Int}
    _clusters_ind_map::Vector{Int}
    _cluster_count_to_ind_map::Vector{Int}
end

function Clusters(n::Int) :: Clusters
    return Clusters(
        MySet(n),
        1,
        fill(1, n),
        collect(1:(n * 2 - 1)),
        collect(1:n)
    )
end

function get_min_count(clusters::Clusters) :: Int
    return clusters._cluster_counts[clusters._min_cluster_count_id]
end

function get_cluster_id(clusters::Clusters, point_ind::Int) :: Int
    return find_root(clusters._set, point_ind)
end

function get_cluster_count(clusters::Clusters, cl_ind::Int) :: Int
    return clusters._cluster_counts[clusters._clusters_ind_map[cl_ind]]
end

# merge clusters, do all internal stuff and return new genie,
function merge_clusters!(clusters::Clusters, cl_ind1::Int, cl_ind2::Int, old_genie::Float64, iteration::Int) :: Float64
    cl_ind1_cluster_count_ind = clusters._clusters_ind_map[cl_ind1]
    cl_ind2_cluster_count_ind = clusters._clusters_ind_map[cl_ind2]
    cl_ind1_cluster_count = clusters._cluster_counts[cl_ind1_cluster_count_ind]
    cl_ind2_cluster_count = clusters._cluster_counts[cl_ind2_cluster_count_ind]
    clusters_sum_count = cl_ind1_cluster_count + cl_ind2_cluster_count

    new_gini = gini_index(clusters._cluster_counts, cl_ind1_cluster_count, cl_ind2_cluster_count, old_genie, iteration)

    lower_count_ind, higher_count_ind = cl_ind1_cluster_count_ind < cl_ind2_cluster_count_ind ? (cl_ind1_cluster_count_ind, cl_ind2_cluster_count_ind) : (cl_ind2_cluster_count_ind, cl_ind1_cluster_count_ind)
    
    last_elem_count_ind = length(clusters._cluster_counts) - iteration + 1
    last_elem_real_id = clusters._cluster_count_to_ind_map[last_elem_count_ind]
    clusters._cluster_counts[higher_count_ind] = clusters._cluster_counts[last_elem_count_ind]
    clusters._clusters_ind_map[last_elem_real_id] = higher_count_ind
    clusters._cluster_count_to_ind_map[higher_count_ind] = last_elem_real_id

    new_cl_ind = union!(clusters._set, cl_ind1, cl_ind2)
    clusters._cluster_counts[lower_count_ind] = clusters_sum_count
    clusters._clusters_ind_map[new_cl_ind] = lower_count_ind
    clusters._cluster_count_to_ind_map[lower_count_ind] = new_cl_ind

    if clusters._min_cluster_count_id == cl_ind1_cluster_count_ind || clusters._min_cluster_count_id == cl_ind2_cluster_count_ind || clusters._min_cluster_count_id == last_elem_count_ind
        clusters._min_cluster_count_id = argmin(@view clusters._cluster_counts[1:(last_elem_count_ind - 1)])
    end
    return new_gini
end

# returns inds of cluster1 and cluster2
function pop_min!(edges::Vector{Edge}, clusters::Clusters) :: Tuple{Int, Int}
    min_cluster_count = get_min_count(clusters)

    # we will walk through queue manually until we find appropriate clusters
    check_item = pop!(edges)
    cl_ind1 = get_cluster_id(clusters, check_item.ind1)
    cl_ind2 = get_cluster_id(clusters, check_item.ind2)
    next_check_id = length(edges)
    while get_cluster_count(clusters, cl_ind1) != min_cluster_count && get_cluster_count(clusters, cl_ind2) != min_cluster_count
        check_item, edges[next_check_id] = edges[next_check_id], check_item
        cl_ind1 = get_cluster_id(clusters, check_item.ind1)
        cl_ind2 = get_cluster_id(clusters, check_item.ind2)
        next_check_id -= 1
    end

    return cl_ind1, cl_ind2
end

# ----------------------------------------------------------------------------------
# general functions

"""
    compute_mst_on_the_fly(X, dist_fn; verbose=false)

Compute the MST of a complete graph represented by `X` columns (objects),
with pairwise distances given by `dist_fn(X[:, i], X[:, j])`.

returns sorted vector of edges, descending order by distance

"""
function _compute_mst_on_the_fly(X::Matrix{Float32}, dist_callable) :: Vector{Edge}
    n = size(X, 2)

    # Large value to represent infinity:
    INF = Float32(Inf)

    Dnn = fill(INF, n)             # Closest distances to MST for each vertex
    Fnn = fill(-1, n)              # For each vertex, the MST vertex it's closest to
    M = collect(1:n)               # Set of vertices not yet in MST (initially all)
    
    # We'll construct an MST of (n-1) edges
    edge_vector = Vector{Edge}()
    sizehint!(edge_vector, n - 1)
    
    lastj = M[1]
    Mleft = n

    @inbounds for _ in 1:(n-1)
        # Update Dnn and Fnn
        current_view = @view X[:, lastj]

        # !!! Multithreading !!!
        # We can use either multithreaded or single threaded version
        # Threads.@threads for i in 2:Mleft
        for i in 2:Mleft
            v = M[i]
            curdist = dist_callable(current_view, @view X[:, v])
            if curdist < Dnn[v]
                Dnn[v] = curdist
                Fnn[v] = lastj
            end
        end

        # Find the vertex in M[2..Mleft] with the minimal Dnn
        bestjpos = 2
        bestj = M[2]
        min_j = Dnn[bestj]
        for i in 3:Mleft
            v = M[i]
            v_j = Dnn[v]
            if v_j < min_j
                bestj = v
                min_j = v_j
                bestjpos = i
            end
        end
        otherj = Fnn[bestj]

        for i in bestjpos:(Mleft-1)
            M[i] = M[i+1]
        end
        
        pq_first, pq_second = otherj < bestj ? (otherj, bestj) : (bestj, otherj)
        item = Edge(pq_first, pq_second, min_j)
        push!(edge_vector, item)

        lastj = bestj
        Mleft -= 1
    end

    # sort edges by distance
    sort!(edge_vector, rev=true)
    return edge_vector
end


function gini_index(cluster_sizes::Vector{Int}, c1::Int, c2::Int, previous_gini::Float64, j::Int) :: Float64
    n = length(cluster_sizes)
    n_j = n - j
    n_j_one = n_j - 1
    added_element = - c1 - c2 + abs(c1 - c2)
    @inbounds for i in 1:(n_j + 1)
        c = cluster_sizes[i]
        added_element += abs(c - c1 - c2) - abs(c - c1) - abs(c - c2)
    end

    gini = (Float64(n_j) / (n_j_one)) * previous_gini + Float64(added_element) / (n * (n_j_one))
    return gini
end

# It calculates full gini index, O(n2)
# function gini_full(cluster_sizes::Vector{Int}, j::Int) :: Float64
#     n = length(cluster_sizes)
#     n_j = n - j
#     gini = 0.0
#     for i in 1:n_j
#         c1 = cluster_sizes[i]
#         for k in (i + 1):n_j
#             c2 = cluster_sizes[k]
#             gini += abs(c1 - c2)
#         end
#     end
#     gini /= n * (n_j - 1)
#     return gini
# end

end
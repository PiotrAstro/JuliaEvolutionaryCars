
mutable struct TreeNode
    left::Union{TreeNode, Nothing}
    right::Union{TreeNode, Nothing}
    elements::Vector{Int}
end

function is_leaf(tree::TreeNode) :: Bool
    return isnothing(tree.left) && isnothing(tree.right)
end

function create_tree(encoded_states::Matrix{Float32}) :: TreeNode
    # change encoded states to pyobject
    encoded_states_py = PyCall.PyObject(encoded_states')

    # get genie clustering
    genie = genieclust.Genie(compute_full_tree=true, verbose=true, affinity="cosine")
    @time genie.fit(encoded_states_py)

    children = collect(Array(genie.children_)')
    # distances_clusters = Array(genie.distances_)

    tree = _create_tree(children, size(encoded_states, 2))

    return tree
end

function _create_tree(children::Matrix{Int}, n_states) :: TreeNode
    clusters = Vector{TreeNode}(undef, n_states*2)
    for i in 1:n_states
        clusters[i] = TreeNode(nothing, nothing, [i])
    end
    last_index = n_states + 1

    for i in 1:size(children, 2)
        left = clusters[children[1, i] + 1]
        right = clusters[children[2, i] + 1]

        clusters[last_index] = TreeNode(left, right, vcat(left.elements, right.elements))
        last_index += 1
    end
    last_index -= 1

    return clusters[last_index]
end
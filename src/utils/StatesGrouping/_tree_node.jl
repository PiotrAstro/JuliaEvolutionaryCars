# ------------------------------------------------------------------------------------------
# TreeNode for hierarchical clustering
# ------------------------------------------------------------------------------------------

struct TreeNode
    left::Union{TreeNode, Nothing}
    right::Union{TreeNode, Nothing}
    elements::Vector{Int}
    priority::Int
end

# sort functions
function Base.isless(a::TreeNode, b::TreeNode) :: Bool
    return a.priority < b.priority
end

function TreeNode(
    left::Union{TreeNode, Nothing},
    right::Union{TreeNode, Nothing},
    elements::Vector{Int}
)
    return TreeNode(left, right, elements, -1)
end


function is_leaf(tree::TreeNode) :: Bool
    return isnothing(tree.left) && isnothing(tree.right)
end


function _create_tree_hclust(merge_matrix::Matrix{Int}) :: TreeNode
    points_n = size(merge_matrix, 1) + 1
    leafs = Vector{TreeNode}(undef, points_n)
    clusters = Vector{TreeNode}(undef, points_n - 1)

    priority = 0
    for i in 1:points_n
        leafs[i] = TreeNode(nothing, nothing, [i], priority)
        priority += 1
    end
    last_index = 1

    for i in 1:size(merge_matrix, 1)
        left_index = merge_matrix[i, 1]
        right_index = merge_matrix[i, 2]

        left = left_index >= 0 ? clusters[left_index] : leafs[-left_index]
        right = right_index >= 0 ? clusters[right_index] : leafs[-right_index]

        clusters[last_index] = TreeNode(left, right, vcat(left.elements, right.elements), priority)
        priority += 1
        last_index += 1
    end
    last_index -= 1

    return clusters[last_index]
end



# ---------------------------------------------------------------------------------------------
# Get levels of the tree
# ---------------------------------------------------------------------------------------------

function get_levels(tree::TreeNode, mode::Symbol=:equal_up) :: Vector{Vector{Vector{Int}}}
    if mode == :equal_up
        return _get_equal_levels(tree, :up)
    elseif mode == :equal_down
        return _get_equal_levels(tree, :down)
    elseif mode == :priority_up
        return _get_priority_levels(tree, :up)
    elseif mode == :priority_down
        return _get_priority_levels(tree, :down)
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end
end

function _get_equal_levels(tree::TreeNode, mode::Symbol=:up) :: Vector{Vector{Vector{Int}}}
    levels = Vector{Vector{Vector{Int}}}()
    current_level = [tree]

    while !isempty(current_level)
        next_level = Vector{TreeNode}()
        
        level = Vector{Vector{Int}}()
        for node in current_level
            push!(level, node.elements)
            if !is_leaf(node)
                push!(next_level, node.left, node.right)
            end
        end
        push!(levels, level)
        current_level = next_level
    end

    if mode == :up
        levels = reverse(levels)
    elseif mode == :down
        # do nothing
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end
    return levels
end


function _get_priority_levels(tree::TreeNode, mode::Symbol=:up) :: Vector{Vector{Vector{Int}}}
    all_nodes = Vector{TreeNode}()
    current_level = [tree]
    while !isempty(current_level)
        next_level = Vector{TreeNode}()
        
        for node in current_level
            push!(all_nodes, node)
            if !is_leaf(node)
                push!(next_level, node.left, node.right)
            end
        end
        current_level = next_level
    end

    sort!(all_nodes)

    levels = [Vector{Vector{Int}}(node) for node in all_nodes]
    if mode == :up
        # nothing
    elseif mode == :down
        reverse!(levels)
    else
        throw(ArgumentError("Unknown mode: $mode"))
    end
    return levels
end
module directionCCL

# include("label_components.jl")

greet() = print("Hello World!")

# INITIAL ADAPTATION, WITHOUT MACROS

import Base: push!

function lc( U, V, dot_th )
  lbls = zeros( Int, size(U) ); 
  lc!( lbls, U, V, dot_th )
  return lbls 
end

function lc!( Albl::AbstractArray{Int,2}, 
              U::AbstractArray{T,2}, 
              V::AbstractArray{T,2},
              dot_th::T, 
              sets=DisjointMinSets() ) where {T}

    uregion = 1:ndims(U)
    offsets = strides(U)[uregion]
    offsets_1 = offsets[1]
    offsets_2 = offsets[2]
    k = 0
    for i_2 in axes(U,2), 
        i_1 in axes(U,1)

        k += 1
        u1 = U[k]
        v1 = V[k]
        label = typemax(Int)

        if i_1 > 1
            dot = u1*U[k-offsets_1] + v1*V[k-offsets_1]
            if dot >= dot_th
                newlabel = Albl[k-offsets_1]
                if label != typemax(Int) && label != newlabel
                    label = union!( sets, label, newlabel ); 
                else
                    label = newlabel
                end
            end
        end

        if i_2 > 1
            dot = u1*U[k-offsets_2] + v1*V[k-offsets_2]
            if dot >= dot_th
                newlabel = Albl[k-offsets_2]
                if label != typemax(Int) && label != newlabel
                    label = union!( sets, label, newlabel ); 
                else
                    label = newlabel
                end
            end
        end

        if label == typemax(Int)
            label = push!(sets)   # there were no neighbors, create a new label
        end

        Albl[k] = label; 
    end

    newlabel = minlabel(sets)
    for i = 1:length(U)
        Albl[i] = newlabel[find_root!(sets, Albl[i])]
    end
    
    return Albl
end

#####

    struct DisjointMinSets
        parents::Vector{Int}
        DisjointMinSets(n::Integer) = new([1:n;])
    end

    DisjointMinSets() = DisjointMinSets(0)

    function find_root!(sets::DisjointMinSets, m::Integer)
        p = sets.parents[m]  # don't use @inbounds here, it might not be safe
        @inbounds if sets.parents[p] != p # marc: when does this even happen?
            sets.parents[m] = p = find_root_unsafe!(sets, p)
        end
        p
    end

    # an unsafe variant of the above
    function find_root_unsafe!(sets::DisjointMinSets, m::Int)
        @inbounds p = sets.parents[m]
        @inbounds if sets.parents[p] != p
            sets.parents[m] = p = find_root_unsafe!(sets, p)
        end
        p
    end

    function union!(sets::DisjointMinSets, m::Integer, n::Integer)
        mp = find_root!(sets, m)
        np = find_root!(sets, n)
        if mp < np
            sets.parents[np] = mp
            # marc: modifies the DisjointMinSets, making the find_root_unsafe necessary
            return mp
        elseif np < mp
            sets.parents[mp] = np
            # marc: modifies the DisjointMinSets, making the find_root_unsafe necessary
            return np
        end
        mp
    end

    function push!(sets::DisjointMinSets)
        m = length(sets.parents) + 1
        push!(sets.parents, m)
        m
    end

    function minlabel(sets::DisjointMinSets)
        out = Vector{Int}(undef, length(sets.parents))
        k = 0
        for i = 1:length(sets.parents)
            if sets.parents[i] == i
                k += 1
            end
            out[i] = k
        end
        out
    end
#

end # module directionCCL

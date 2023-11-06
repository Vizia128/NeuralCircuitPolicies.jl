using Flux, LinearAlgebra, ComponentArrays, StaticArrays
import Base.split

struct WiredCFCCell
    input_size::Integer
    layer_sizes::Vector
    cells
end

function WiredCFCCell()
    
end

function split_array(array::AbstractArray, layer_sizes::Base.AbstractVecOrTuple{Integer}; dim::Integer = 1)
    end_idxs = cumsum(layer_sizes)
    start_idxs = [1; end_idxs[1:end-1] .+ 1]

    return [selectdim(array, dim, i:j) for (i, j) in zip(start_idxs, end_idxs)]
end

function (model::WiredCFCCell)(input, hx, timespans)
    hidden_states = split_array(hx, model.layer_sizes)

    new_hidden_state = []
    for (i, hidden_state) in enumerate(hidden_states)
        input, _ = model.cells[i](input, hidden_state, timespans)
        push!(new_hidden_state, input)
    end
    new_hidden_state = cat(new_hidden_state; dims=2)

    return input, new_hidden_state
end

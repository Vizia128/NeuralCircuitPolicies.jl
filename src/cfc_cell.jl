using Flux, LinearAlgebra


struct CfCCell
    input_size::Integer
    hidden_size::Integer
    ff1::Dense
    ff2::Dense
    time_a::Dense
    time_b::Dense
end

function CfCCell(input_size::Integer, hidden_size::Integer)
    cat_shape = input_size + hidden_size

    ff1 = Dense(cat_shape => hidden_size)
    ff2 = Dense(cat_shape => hidden_size)
    time_a = Dense(cat_shape => hidden_size)
    time_b = Dense(cat_shape => hidden_size)

    CfCCell(input_size, hidden_size, ff1, ff2, time_a, time_b)
end

function (model::CfCCell)(input, hx, timespans)
    x = cat(input, hx; dims=2)

    ff1 = model.ff1(x)
    ff2 = model.ff2(x)
    t_a = model.time_a(x)
    t_b = model.time_b(x)

    t_interp = sigmoid(t_a * timespans + t_b)
    new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2

    return new_hidden
end

Flux.@functor CfCCell



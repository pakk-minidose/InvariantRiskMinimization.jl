using Flux, ValueHistories, Random, CUDA, BSON
using Statistics: mean
using LinearAlgebra

"""
    accuracy(m, X, Y)

Compute prediction accuracy of labels `Y` for classifier `m` on input `X`.

The labels are expected to be from the set {-1,1}.
"""
function accuracy(m, X, Y)
    Ŷ = sign.(m(X))
    return mean(Y .== Ŷ)
end

"""
    penalty(Φ, X, Y; T=eltype(Φ))

Compute the IRM penalty term for the MSE loss.

The function uses analytically derived formula of the derivative for better numerical stability.
"""
function penalty(Φ, X, Y; T=eltype(Φ))
    @assert ndims(X)==2
    @assert size(X,2) == length(Y)
    nsamples = size(X,2)
    Φt = permutedims(Φ)
    z = Φt*X
    v = dot(z-Y, z)
    coefficient = T(2/nsamples)
    penalty_value = (coefficient*v)^2
    return penalty_value
end

function make_minibatch(dataset, mbatchsize, rng)
    dataset_batch = map(dataset) do dset
        Ix = rand(rng, 1:size(dset.X,2), mbatchsize)
        return (X=cu(dset.X[:,Ix]), Y = cu(dset.Y[:,Ix]))
    end
    return dataset_batch
end

function irm(dataset, niter, λ, η; rng=Xoshiro(rand(UInt32)), mbatchsize=nothing, val_dataset=nothing, init_coeff=1f-1, with_bias=false, early_stop=false, filename_salt=nothing)
    if early_stop && (isnothing(val_dataset) || isnothing(filename_salt))
        error("Invalid combination of arguments: early_stop cannot be set to true if val_dataset=nothing or filename_salt=nothing")
    end
    
    history = MVHistory()

    if isnothing(mbatchsize) #transfer whole datset to GPU
        cuda_dataset = map(dataset) do dset
            return (X=cu(dset.X), Y=cu(dset.Y))
        end
    end

    indims = size(dataset[1].X,1)#assuming each column is an observation
    Φ = initΦ(indims, with_bias, init_coeff, rng)
    m(x) = Φ'*x

    mse_loss(x, y) = Flux.mse(m(x), y)
    lossfun(_dataset) = sum(_dataset) do dset
        mse_loss(dset.X, dset.Y) + λ*penalty(Φ, dset.X, dset.Y)
    end
    lossfun_with_cu(_dataset) = sum(_dataset) do dset
        X = cu(dset.X)
        Y = cu(dset.Y)
        return mse_loss(X, Y) + λ*penalty(Φ, X, Y)
    end
    
    if early_stop #these variables and functions are needed only if early stopping is enabled
        objective_function(acc_vector) = minimum(acc_vector)
        best_objective_value = -Inf
        filename = string("irm_autosave_", getpid(), "_", Threads.threadid(), "_", filename_salt, ".bson")
    end

    ps = Flux.params(Φ)
    opt = Flux.Adam(η)

    for iter_ix in 1:niter
        if isnothing(mbatchsize)
            gs, loss_value = compute_irm_gs_fullbatch(ps, cuda_dataset, lossfun)
        else
            gs, loss_value = compute_irm_gs_minibatch(dataset, mbatchsize, rng, ps, lossfun)
        end
        push!(history, :loss, iter_ix, loss_value)
        Flux.update!(opt, ps, gs)
        if iter_ix%10==0
            @show iter_ix loss_value
            if !isnothing(val_dataset)
                val_acc = log_irm_history!(history, iter_ix, Φ, m, λ, gs, ps, val_dataset, lossfun_with_cu)
            end
            if early_stop
                best_objective_value = early_stop_irm(objective_function, best_objective_value, val_acc, Φ, filename)
            end
        end
    end

    if early_stop
        cpu_Φ = BSON.load(filename)[:Φ]
        rm(filename)
    else
        cpu_Φ = cpu(Φ)
    end
    cpu_m(x) = cpu_Φ'*x

    return cpu_m, cpu_Φ, history
end

function early_stop_irm(objective_function, best_objective_value, val_acc, gpuΦ, filename)
    objective_value = objective_function(val_acc)
    if objective_value >= best_objective_value
        Φ = cpu(gpuΦ)
        BSON.@save filename Φ
        return objective_value
    else
        return best_objective_value
    end
end

function compute_irm_gs_fullbatch(ps, cuda_dataset, lossfun)
    gs = gradient(ps) do
        lossfun(cuda_dataset)
    end
    loss_value = lossfun(cuda_dataset)
    return gs, loss_value
end

function compute_irm_gs_minibatch(dataset, mbatchsize, rng, ps, lossfun)
    cuda_batch = make_minibatch(dataset, mbatchsize, rng)
    gs = gradient(ps) do
        lossfun(cuda_batch)
    end
    loss_value = lossfun(cuda_batch)
    return gs, loss_value
end

function log_irm_history!(history, iter_ix, Φ, m, λ, gs, ps, val_dataset, lossfun_with_cu)
    val_loss_value = lossfun_with_cu(val_dataset)
    push!(history, :val_loss, iter_ix, val_loss_value)
    val_acc = map(val_dataset) do dset
        accuracy(m, cu(dset.X), cu(dset.Y))
    end
    push!(history, :val_acc, iter_ix, val_acc)
    val_penalty = map(val_dataset) do dset
        λ*penalty(Φ, cu(dset.X), cu(dset.Y))
    end
    push!(history, :val_penalty, iter_ix, val_penalty)
    sum_gs = sum(map(p->sum(abs2, gs[p]), ps))
    push!(history, :sum_gs, iter_ix, sum_gs)
    sum_Φ = sum(abs2, Φ)
    push!(history, :sum_Phi, iter_ix, sum_Φ)
    return val_acc
end

function initΦ(indims, with_bias, init_coeff, rng)
    if with_bias
        cpuΦ = init_coeff .*randn(rng, Float32, indims)
        cpuΦ[end] = 0f0 #set bias to zero
        Φ = cu(cpuΦ)
    else
        Φ = cu(init_coeff .*randn(rng, Float32, indims))
    end
end
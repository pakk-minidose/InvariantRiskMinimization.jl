using Flux, ValueHistories, Random, BSON
using Statistics: mean
using LinearAlgebra

"""
    irm(envdatasets, niter, λ, η; <keyword arguments>)

Train a linear classifier using the Invariant Risk Minimization [^1] method.

The implementation assumes MSE loss and binary classification with labels
from the set {-1, 1}. ADAM optimizes is used to update the classifier parameters.

# Arguments
- `envdatasets`: vector of named tuples with field names `X` and `Y`. Each element
    corresponds to one environment. `X` and `Y` correspond to the input and labels.
- `niter`: number of iterations of the optimization.
- `λ`: multiplies the value of the penalty term in the loss function
- `η`: optimization step size

# Keyword arguments
- `rng=Xoshiro`: used to pass a random number to the function generator, e.g. in order to
    obtain reproducible results.
- `mbatchsize=nothing`: if `nothing`, full batch training is performed. Otherwise a vector
    of minibatches each of size `mbatchsize` is sampled in each iteration of training.
- `val_envdatasets=nothing`: used both to pass the validation datasets and enable extended
    logging. If `nothing`, the extended the logging is disabled. Otherwise, the extended
    logging is enabled and `val_envdatasets` needs to have the same form as `envdatasets`.
    The extended logging is performed once per 10 training steps.
- `initcoeff=1f-1`: standard deviation of the normal distribution used to initialize the
    model parameters.
- `withbias=false`: if set to `true`, the last feature of the input data is expected to be
    set to 1 in order to allow a classifier with bias.
- `earlystop=false`: if set to `true`, uses `val_dataset` for early stopping. The
    val_dataset cannot be `nothing` in this case. The early stopping criterion maximizes
    the minimal accuracy across environments. The early stopping is evaluated once per
    10 training steps.
- `filename_salt=nothing`: used in the temporary file for early stopping to allow for
    non-conflicting filenames.

# Returned values
- `Φ`: parametric vector of the resulting classifier. The classifier is defined
    as `m(x)=Φ'*x`.
- `history`: `MVHistory` with values logged during training. 

[^1]: ARJOVSKY, Martin, et al. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.
"""
function irm(envdatasets, niter, λ, η; rng=Xoshiro(rand(UInt32)), mbatchsize=nothing,
    val_envdatasets=nothing, initcoeff=1f-1, withbias=false,
    earlystop=false, filename_salt=nothing)
    
    if earlystop && (isnothing(val_envdatasets) || isnothing(filename_salt))
        error("Invalid combination of arguments: earlystop cannot be set to true if val_envdatasets=nothing or filename_salt=nothing")
    end
    
    history = MVHistory()

    if isnothing(mbatchsize) #full batch training, transfer whole train datset to GPU
        cuda_dataset = map(envdatasets) do dset
            return (X=gpu(dset.X), Y=gpu(dset.Y))
        end
    end

    indims = size(envdatasets[1].X,1)#assuming each column is an observation
    Φ = initΦ(indims, withbias, initcoeff, rng)
    m(x) = Φ'*x

    mse_loss(x, y) = Flux.mse(m(x), y)
    lossfun(_dataset) = sum(_dataset) do dset
        mse_loss(dset.X, dset.Y) + λ*penalty(Φ, dset.X, dset.Y)
    end
    lossfun_with_cu(_dataset) = sum(_dataset) do dset
        X = gpu(dset.X)
        Y = gpu(dset.Y)
        return mse_loss(X, Y) + λ*penalty(Φ, X, Y)
    end
    
    if earlystop #these variables and functions are needed only if early stopping is enabled
        objective_function(acc_vector) = minimum(acc_vector)
        best_objective_value = -Inf
        filename = string("irm_autosave_", getpid(), "_", Threads.threadid(), "_", filename_salt, ".bson")
    end

    ps = Flux.params(Φ)
    opt = Flux.Adam(η)

    for iter_ix in 1:niter #training loop
        if isnothing(mbatchsize)
            gs, loss_value = compute_irm_gs_fullbatch(ps, cuda_dataset, lossfun)
        else
            gs, loss_value = compute_irm_gs_minibatch(envdatasets, mbatchsize, rng, ps, lossfun)
        end
        push!(history, :loss, iter_ix, loss_value)
        Flux.update!(opt, ps, gs)
        if iter_ix%10==0 #to improve performance, do not perform logging and early stopping each step
            @show iter_ix loss_value
            if !isnothing(val_envdatasets)
                val_acc = log_irm_history!(history, iter_ix, Φ, m, λ, gs, ps, val_envdatasets, lossfun_with_cu)
            end
            if earlystop
                best_objective_value = early_stop_irm(objective_function, best_objective_value, val_acc, Φ, filename)
            end
        end
    end

    if earlystop #load early stop result
        cpu_Φ = BSON.load(filename)[:Φ] #the result is already transfered to cpu
        rm(filename)
    else
        #if we do not load the early stop result, we need to transfer Φ to cpu
        cpu_Φ = cpu(Φ)
    end

    return cpu_Φ, history
end

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

The function uses analytically derived formula of the derivative for better numerical
stability.
"""
function penalty(Φ, X, Y; T=eltype(Φ))
    @assert ndims(X)==2
    @assert size(X,2) == length(Y)
    nsamples = size(X,2)
    Φt = permutedims(Φ)
    z = Φt*X
    v = dot(z-Y, z)
    coefficient = T(2/nsamples)
    penaltyvalue = (coefficient*v)^2
    return penaltyvalue
end

"""
    makeminibatch(envdatasets, half_mbatchsize, rng)

Sample vector of minibatches from the vector of environment datasets.

The size of each minibatch is `mbatchsize`. Random number generator `rng`
is used in the process.
"""
function makeminibatch(envdatasets, mbatchsize, rng)
    dataset_batch = map(envdatasets) do dset
        Ix = rand(rng, 1:size(dset.X,2), mbatchsize)
        return (X=gpu(dset.X[:,Ix]), Y = gpu(dset.Y[:,Ix]))
    end
    return dataset_batch
end

"""
Evaluate early stopping objective, save if objective value improved.
"""
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

"""
Compute gradient and loss value on the whole training dataset.
"""
function compute_irm_gs_fullbatch(ps, cuda_dataset, lossfun)
    gs = gradient(ps) do
        lossfun(cuda_dataset)
    end
    loss_value = lossfun(cuda_dataset)
    return gs, loss_value
end

"""
Compute gradient and loss value on the minibatch.
"""
function compute_irm_gs_minibatch(dataset, mbatchsize, rng, ps, lossfun)
    cuda_batch = makeminibatch(dataset, mbatchsize, rng)
    gs = gradient(ps) do
        lossfun(cuda_batch)
    end
    loss_value = lossfun(cuda_batch)
    return gs, loss_value
end

function log_irm_history!(history, iter_ix, Φ, m, λ, gs, ps, val_envdatasets, lossfun_with_cu)
    val_loss_value = lossfun_with_cu(val_envdatasets)
    push!(history, :val_loss, iter_ix, val_loss_value)

    val_acc = map(val_envdatasets) do dset
        accuracy(m, gpu(dset.X), gpu(dset.Y))
    end
    push!(history, :val_acc, iter_ix, val_acc)

    val_penalty = map(val_envdatasets) do dset
        λ*penalty(Φ, gpu(dset.X), gpu(dset.Y))
    end
    push!(history, :val_penalty, iter_ix, val_penalty)

    sum_gs = sum(map(p->sum(abs2, gs[p]), ps))
    push!(history, :sum_gs, iter_ix, sum_gs)

    sum_Φ = sum(abs2, Φ)
    push!(history, :sum_Phi, iter_ix, sum_Φ)
    
    return val_acc
end

function initΦ(indims, withbias, initcoeff, rng)
    if withbias
        cpuΦ = initcoeff .*randn(rng, Float32, indims)
        cpuΦ[end] = 0f0 #set bias to zero
        Φ = gpu(cpuΦ)
    else
        Φ = gpu(initcoeff .*randn(rng, Float32, indims))
    end
end
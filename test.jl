using Pkg; Pkg.activate(".")
using Test, BSON, StableRNGs

@testset "IRM tests" begin
    include("irm.jl")
    @testset "Penalty" begin
        function loadTestValues(filename)
            tmp = BSON.load("./test/$filename")
            return tmp[:Φ], tmp[:X], tmp[:Y], tmp[:result]
        end
        @test_throws AssertionError penalty(nothing, ones(Float64, 2, 999), ones(Float64, 1, 1000))
        @test_throws AssertionError penalty(nothing, ones(Float64, 3, 3, 3), nothing)
        Φ, X, Y, result = loadTestValues("test1.bson")
        @test penalty(Φ, X, Y) ≈ result #test that penalty gives the same results
        Φ, X, Y, result = loadTestValues("test2.bson")
        @test penalty(Φ, X, Y) ≈ result #test that penalty gives the same results
        @test typeof(result) == Float32
    end

    dataset = BSON.load("./test/testdataset.bson")[:dataset]
    @testset "makeMinibatch" begin
        mbatchsize = 128
        dataset_batch = makeminibatch(dataset, mbatchsize, StableRNG(814299091))
        @test length(dataset_batch) == 2
        @test size(dataset_batch[1].X) == (2, mbatchsize)
        @test size(dataset_batch[1].Y) == (1, mbatchsize)
        trivial_dataset = [(X=ones(2,1000), Y=2*ones(1,1000));(X=zeros(2,1000),Y=-1*ones(1,1000))]
        dataset_batch = makeminibatch(trivial_dataset, mbatchsize, StableRNG(144090918))
        @test all(dataset_batch[1].X .== 1)
        @test all(dataset_batch[1].Y .== 2)
        @test all(dataset_batch[2].X .== 0)
        @test all(dataset_batch[2].Y .== -1)
    end
    @testset "Full batch IRM" begin
        Φ, history = irm(dataset, 10, 200, 1e-3, rng=StableRNG(990912203), initcoeff=1f0)
        @test all(Φ .≈ [1.4610121f0;1.0043464f0]) #test that IRM gives the same results
    end
    @testset "Minibatch IRM" begin
        Φ, history = irm(dataset, 10, 100, 1e-3, rng=StableRNG(990912203),mbatchsize=128, initcoeff=1f0)
        @test all(Φ .≈ [1.4611939f0;1.004548f0]) #test that IRM gives the same results
    end  
    @testset "Validation dataset" begin
        Φ, history = irm(dataset, 10, 100, 1e-3, rng=StableRNG(990912203),mbatchsize=128, val_envdatasets=dataset, initcoeff=1f0)
        @test all(Φ .≈ [1.4611939f0;1.004548f0]) #test that IRM gives the same results
    end
end
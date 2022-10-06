using Pkg; Pkg.activate(".")
using Test, Random, BSON, LinearAlgebra, StableRNGs, DataFrames

@testset "Basic IRM tests" begin
    include("irm.jl")
    @testset "Penalty" begin
        function loadTestValues(filename)
            tmp = BSON.load("./test/$filename")
            return tmp[:Φ], tmp[:X], tmp[:Y], tmp[:result]
        end
        @test_throws AssertionError penalty(nothing, ones(Float64, 2, 999), ones(Float64, 1, 1000))
        @test_throws AssertionError penalty(nothing, ones(Float64, 3, 3, 3), nothing)
        Φ, X, Y, result = loadTestValues("test1.bson")
        @test penalty(Φ, X, Y) ≈ result
        Φ, X, Y, result = loadTestValues("test2.bson")
        @test penalty(Φ, X, Y) ≈ result
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
        m, Φ, history = irm(dataset, 10, 200, 1e-3, rng=StableRNG(990912203), init_coeff=1f0)
        @test all(Φ .≈ [1.4610121f0;1.0043464f0])
    end
    @testset "Minibatch IRM" begin
        m, Φ, history = irm(dataset, 10, 100, 1e-3, rng=StableRNG(990912203),mbatchsize=128, init_coeff=1f0)
        @test all(Φ .≈ [1.4611939f0;1.004548f0])
    end  
    @testset "Validation dataset" begin
        m, Φ, history = irm(dataset, 10, 100, 1e-3, rng=StableRNG(990912203),mbatchsize=128, val_dataset=dataset, init_coeff=1f0)
        @test all(Φ .≈ [1.4611939f0;1.004548f0])
    end
end

@testset "loadCSMData.jl" begin
    include("loadCSMData.jl")
    @testset "Basic binary files loading" begin
        C, S, iso = loadCameraFeatures("testdataset_CSM", "./test")
        @test all(iso[1:100] .== 100)
        @test all(iso[101:end] .== 200)
        @test size(C) == (22510, 200)
        @test size(S) == (22510, 200)
        @test C[1,1]==0.23867039323748185 && C[15411,99]==0.6086129007570938 && C[2711,181]==0.14717184447143228
        @test S[1,1]==0.6789685275518987 && S[14411,89]==0.13306657887521123 && S[13711,172]==0.0844007110693743
    end
    @testset "makeVectorOfDatasets" begin
        df = BSON.load("./test/testdataset_CSM2.bson")[:df]
        dta1 = makeVectorOfDatasets(df, 1, type=:train, train_rows=1:1)
        X1 = dta1[1].X
        Y1 = dta1[1].Y
        @test all(Y1[1:70] .== 1.0)
        @test all(Y1[71:end] .== -1.0)
        Ix = [CartesianIndex(100,1); CartesianIndex(22184,111); CartesianIndex(10052,54)]
        @test X1[Ix] == [0.7073894f0;0.50549364f0;0.7053786f0]
        @test size(X1) == (22511,140)
        @test all(X1[end,:] .== 1)
        dta2 = makeVectorOfDatasets(df, 1, type=:train, train_rows=1:1,add_bias_feature=false)
        X2 = dta2[1].X
        Y2 = dta2[1].Y
        @test Y2 == Y1
        @test X2 == X1[1:end-1,:]
        @test size(X2) == (22510,140)
    end
end

@testset "DCTR features" begin
    include("loadDCTRData.jl")
    cs_ix = 1
    path = "/home/sepakdom/DCTR_dataset/DCTRfeatures"
    A1 = getData([cs_ix], path, 0.4, "train")[1]
    function loadTestValues(filename)
        tmp = BSON.load("./test/$filename")
        return tmp[:test_indices_X], tmp[:test_indices_Y], tmp[:values_X], tmp[:values_Y]
    end
    test_indices_X, test_indices_Y, values_X, values_Y = loadTestValues("DCTRload.bson")
    @test A1.X[test_indices_X] == values_X
    @test A1.Y[test_indices_Y] == values_Y

    A2 = getData([cs_ix], path, 0.4, "train", true)[1]
    @test A1.X == A2.X[1:end-1,:]
    @test A1.Y == A2.Y
    @test all(A2.X[end,:] .== 1)
end



@testset "utilities" begin
    @testset "iselequal()" begin
        @test TensorOperations._iselequal([1,2],(3,4), [1,2,3,4]) == true
        @test TensorOperations._iselequal((1,2),(3,4), (1,2,3,4)) == true
        @test TensorOperations._iselequal((1,2),(3), (1,2,3,4)) == false
        @test TensorOperations._iselequal((1,2),(3.0), (1,2,3,4)) == false
        @test TensorOperations._iselequal((1,2),(3,5), (1,2,3,4)) == false
    end

    @testset "_size()" begin
        A = randn(4,5,6)
        @test TensorOperations._size(A) == (4, 5, 6)
        @test TensorOperations._size(A, tuple()) == tuple()
        @test TensorOperations._size(A, 2) == (5,)
        @test TensorOperations._size(A, (2,)) == (5,)
        @test TensorOperations._size(A, (2,1)) == (5,4)

        @test TensorOperations._size(A, []) == tuple()
        @test TensorOperations._size(A, [1]) == (4,)
        @test TensorOperations._size(A, [1, 3]) == (4,6)
    end
end

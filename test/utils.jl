facts("utilities") do
    context("iselequal()") do
        @fact TensorOperations._iselequal([1,2],(3,4), [1,2,3,4]) --> true
        @fact TensorOperations._iselequal((1,2),(3,4), (1,2,3,4)) --> true
        @fact TensorOperations._iselequal((1,2),(3), (1,2,3,4)) --> false
        @fact TensorOperations._iselequal((1,2),(3.0), (1,2,3,4)) --> false
        @fact TensorOperations._iselequal((1,2),(3,5), (1,2,3,4)) --> false
    end

    context("_size()") do
        A = randn(4,5,6)
        @fact TensorOperations._size(A) --> (4, 5, 6)
        @fact TensorOperations._size(A, tuple()) --> tuple()
        @fact TensorOperations._size(A, 2) --> (5,)
        @fact TensorOperations._size(A, (2,)) --> (5,)
        @fact TensorOperations._size(A, (2,1)) --> (5,4)

        @fact TensorOperations._size(A, []) --> tuple()
        @fact TensorOperations._size(A, [1]) --> (4,)
        @fact TensorOperations._size(A, [1, 3]) --> (4,6)
    end
end

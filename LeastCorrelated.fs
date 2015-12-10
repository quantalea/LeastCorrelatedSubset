module LeastCorrelated

open System
open Alea.CUDA
open Alea.CUDA.Compilation
open Alea.CUDA.Utilities
open Alea.CUDA.Unbound

open Binomial

/// Convert to array of array to 2d array.
let toMatrix (A:float[][]) = 
    Array2D.init A.Length A.[0].Length (fun i j -> A.[i].[j])

/// Size of lower triangular part = n*(n+1)/2 - n.
[<ReflectedDefinition>]
let size n = n*(n-1)/2     

/// Packed offset for i > j, lower triangular part only, as diagonal is all 1 and therefore redundant.
[<ReflectedDefinition>]
let offset i j = size i + j

/// Pack lower triangular part.
let lowerTriangularPacked (A:float[][]) = 
    A |> Array.mapi (fun i row -> Array.sub row 0 i) |> Array.concat

/// Unpack to 2d array.
let unpack2D n (A:float[]) =
    Array2D.init n n (fun i j -> if i > j then A.[offset i j] else if i < j then A.[offset j i] else 1.0)

// Unpack to array of array.
let unpack n (A:float[]) =
    Array.init n (fun i ->
        Array.init n (fun j -> 
            if i > j then A.[offset i j] else if i < j then A.[offset j i] else 1.0
        ))

/// Selection indices must be sorted in inreasing order.
[<ReflectedDefinition>]
let subMatixPacked (matrix: int -> float) (subMatrix: int -> float -> unit) (selection:int -> int) selLength =
    let mutable l = 0 
    for i = 0 to selLength - 1 do
        for j = 0 to i - 1 do
            //printfn "(%d, %d) -> (%d, %d)" i j s.[i] s.[j]
            let offset = offset (selection i) (selection j)
            matrix offset |> subMatrix l
            l <- l + 1

[<ReflectedDefinition>]
let private square a = a*a

/// Distance to identity. 
[<ReflectedDefinition>]
let distToIdentity n (A: int -> float) =
    let mutable dist = 0.0
    for i = 0 to n-1 do
        dist <- dist + square(A i) 
    sqrt dist

module Cpu =

    /// A only holds lower triangular part, Frobenius distance to identity 
    /// is just the sqrt of the sum of squares.
    let distToIdentity (A:float[]) = 
        distToIdentity A.Length (fun i -> A.[i])

    let subMatixPacked (A:float[]) (selection:int[]) =
        let subMatrix = Array.zeroCreate (size selection.Length)
        subMatixPacked (fun i -> A.[i]) (fun i v -> subMatrix.[i] <- v) (fun i -> selection.[i]) selection.Length
        subMatrix

    /// Given n random variables with correlation matrix C in packed format,
    /// calculates the distance to the identity for each possible selection of k out of n. This only
    /// works for small values of n and k, i.e. where (choose n k) is in the integer range.
    let allDistToIdentity (C : float[]) n k =
        let cnk = Binomial.choose n k
        Array.init (int cnk) (fun i -> 
            let selection = Binomial.Cpu.subset n k (uint64 i)
            i, subMatixPacked C selection |> distToIdentity
        )

    /// Given n random variables with correlation matrix C in packed format,
    /// finds the k least correlated subgroup.
    let leastCorrelated (C : float[]) n k =
        let best = allDistToIdentity C n k |> Array.minBy snd
        Binomial.Cpu.subset n k (fst best |> uint64)

    /// Convenience function that returns the best combination and the corresponding correlation matrix.
    let leastCorrelatedMatrix (A:float[][]) k =
        let n = A.Length
        let A' = lowerTriangularPacked A
        let best = leastCorrelated A' n k
        best, subMatixPacked A' best |> unpack k

module Gpu =

    let divup (num:uint64) (den:uint64) = (num + den - 1UL) / den

    let worker = Worker.Default

    [<Struct>]
    type IndexAndValue =
        val Index : uint64
        val Value : float
        [<ReflectedDefinition>]
        new(i, v) = { Index = i; Value = v }
        override this.ToString() = sprintf "(%d, %f)" this.Index this.Value

        [<ReflectedDefinition>]
        static member Min (a:IndexAndValue) (b:IndexAndValue) =
            if a.Value < b.Value then a else b

    [<ReflectedDefinition>]
    let transform (inputs:deviceptr<float>) (outputs:deviceptr<IndexAndValue>) (m0:uint64) (n:int) =
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x
        let mutable i = start
        while i < n do
            outputs.[i] <- IndexAndValue(m0 + uint64 i, inputs.[i])
            i <- i + stride

    let minReductionModule = new DeviceReduceModule<IndexAndValue>(GPUModuleTarget.Worker(worker), <@ IndexAndValue.Min @>)

    /// Kernel to calculate distance to identity using device memory for temporary storage.
    [<ReflectedDefinition>]
    let distToIdentityKernelDevice (n:int) (k:int) (cnk:int) (ns:int) (C:deviceptr<float>) (dist:deviceptr<float>) (selection:deviceptr<int>) (subMatrix:deviceptr<float>) =
        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x

        let selection = selection + start * k
        let subMatrix = subMatrix + start * ns

        let mutable m = start  

        while m < cnk do
            
            subset (fun i -> selection.[i]) (fun i v -> selection.[i] <- v) n k (uint64 m)
            subMatixPacked (fun i -> C.[i]) (fun i v -> subMatrix.[i] <- v) (fun i -> selection.[i]) k

            dist.[m] <- distToIdentity ns (fun i -> subMatrix.[i])

            m <- m + stride

    /// The length of the local array must be a GPU compile time constant.
    let Kmax = 10

    /// Using shared memory for C and subMatrix, thread local memory for selection and returning directly an array of float.  
    [<ReflectedDefinition>]
    let distToIdentityKernelCSharedLocal (n:int) (k:int) (cnk:int) (ns:int) (C:deviceptr<float>) (dist:deviceptr<float>) (subMatrix:deviceptr<float>) =
        let sharedSize = size n
        let sharedC = __shared__.ExternArray<float>() |> __array_to_ptr

        let mutable i = threadIdx.x
        while i < sharedSize do
            sharedC.[i] <- C.[i]
            i <- i + blockDim.x

        __syncthreads()

        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x

        let selection = __local__.Array<int>(Kmax)
        let subMatrix = subMatrix + start * ns

        let mutable m = start  

        while m < cnk do
            
            subset (fun i -> selection.[i]) (fun i v -> selection.[i] <- v) n k (uint64 m)
            subMatixPacked (fun i -> sharedC.[i]) (fun i v -> subMatrix.[i] <- v) (fun i -> selection.[i]) k

            dist.[m] <- distToIdentity ns (fun i -> subMatrix.[i])

            m <- m + stride

    /// Using shared memory for C, selection and subMatrix and returning directly the distances as a float array.  
    [<ReflectedDefinition>]
    let distToIdentityKernelShared (m0:uint64) (partitionSize:int) (n:int) (k:int) (ns:int) (C:deviceptr<float>) (dist:deviceptr<float>) =
        let sizeC = size n
        let sizeSelection = blockDim.x * k
        let shared = __shared__.ExternArray<byte>() |> __array_to_ptr
        let sharedC = shared.Reinterpret<float>()
        let selection = (sharedC + sizeC).Reinterpret<int>()
        let subMatrix = (selection + sizeSelection).Reinterpret<float>()

        let mutable i = threadIdx.x
        while i < sizeC do
            sharedC.[i] <- C.[i]
            i <- i + blockDim.x

        __syncthreads()

        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x

        let selection = selection + threadIdx.x * k
        let subMatrix = subMatrix + threadIdx.x * ns

        let mutable m = start  

        while m < partitionSize do
            
            subset (fun i -> selection.[i]) (fun i v -> selection.[i] <- v) n k (m0 + uint64 m)
            subMatixPacked (fun i -> sharedC.[i]) (fun i v -> subMatrix.[i] <- v) (fun i -> selection.[i]) k

            dist.[m] <- distToIdentity ns (fun i -> subMatrix.[i])

            m <- m + stride

    /// Using shared memory for C, selection and subMatrix and returning directly an array of IndexAndValue.  
    [<ReflectedDefinition>]
    let distToIdentityKernelSharedIndexAndValue (m0:uint64) (partitionSize:int) (n:int) (k:int) (ns:int) (C:deviceptr<float>) (dist:deviceptr<IndexAndValue>) =
        let sizeC = size n
        let sizeSelection = blockDim.x * k
        let shared = __shared__.ExternArray<byte>() |> __array_to_ptr
        let sharedC = shared.Reinterpret<float>()
        let selection = (sharedC + sizeC).Reinterpret<int>()
        let subMatrix = (selection + sizeSelection).Reinterpret<float>()

        let mutable i = threadIdx.x
        while i < sizeC do
            sharedC.[i] <- C.[i]
            i <- i + blockDim.x

        __syncthreads()

        let start = blockIdx.x * blockDim.x + threadIdx.x
        let stride = gridDim.x * blockDim.x

        let selection = selection + threadIdx.x * k
        let subMatrix = subMatrix + threadIdx.x * ns

        let mutable m = start  

        while m < partitionSize do
            
            subset (fun i -> selection.[i]) (fun i v -> selection.[i] <- v) n k (m0 + uint64 m)
            subMatixPacked (fun i -> sharedC.[i]) (fun i v -> subMatrix.[i] <- v) (fun i -> selection.[i]) k

            dist.[m] <- IndexAndValue(m0 + uint64 m, distToIdentity ns (fun i -> subMatrix.[i]))

            m <- m + stride

    let allDistToIdentityDevice (C : float[]) n k =
        let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT
        let cnk = choose n k |> int
        let ns = size k
        let blockSize = 128
        let gridSize = 8 * numSm
        use dC = worker.Malloc(C)
        use dDist = worker.Malloc(cnk) 
        use dSelection = worker.Malloc(gridSize * blockSize * k)
        use dSubMatrix = worker.Malloc(gridSize * blockSize * ns)

        let lp = new LaunchParam(gridSize, blockSize)
        worker.Launch <@ distToIdentityKernelDevice @> lp n k cnk ns dC.Ptr dDist.Ptr dSelection.Ptr dSubMatrix.Ptr
        dDist.Gather()    
        
    let allDistToIdentityDeviceCSharedLocal (C : float[]) n k =
        let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT
        let cnk = choose n k |> int
        let ns = size k
        let blockSize = 128
        let gridSize = 8 * numSm
        let sharedSize = __sizeof<float>() * size n 
        use dC = worker.Malloc(C)
        use dDist = worker.Malloc(cnk) 
        use dSubMatrix = worker.Malloc(gridSize * blockSize * ns)

        let lp = new LaunchParam(gridSize, blockSize, sharedSize)
        worker.Launch <@ distToIdentityKernelCSharedLocal @> lp n k cnk ns dC.Ptr dDist.Ptr dSubMatrix.Ptr
        dDist.Gather()    
  
    let allDistToIdentityDeviceShared (C : float[]) n k =
        let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT
        let maxSharedMem = worker.Device.Attributes.MAX_SHARED_MEMORY_PER_BLOCK
        let ns = size k
        let cnk = choose n k
        let blockSize = 128
        let gridSize = 8 * numSm
        let sharedSize = 
            __sizeof<float>() * size n + 
            blockSize * __sizeof<int>() * k + 
            blockSize * __sizeof<float>() * ns

        if sharedSize > maxSharedMem then 
            failwithf "too much shared memory required: max shared mem = %d, required shared memory size = %d" maxSharedMem sharedSize

        let m0 = 0UL
        let mSteps = cnk |> int
        use dC = worker.Malloc(C)
        use dDist = worker.Malloc(mSteps) 

        let lp = new LaunchParam(gridSize, blockSize, sharedSize)
        worker.Launch <@ distToIdentityKernelShared @> lp m0 mSteps n k ns dC.Ptr dDist.Ptr 
        dDist.Gather()  
               
    /// Find least correlated subset with all temporaries in shared memory.
    /// Using kernel `distToIdentityKernelShared` which requires additional transform step. 
    let leastCorrelatedWithTransform (partitionSize:int) (C : float[]) n k =
        let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT
        let maxSharedMem = worker.Device.Attributes.MAX_SHARED_MEMORY_PER_BLOCK
        let ns = size k
        let cnk = choose n k
        let blockSize = 128
        let gridSize = 8 * numSm
        let sharedSize = 
            __sizeof<float>() * size n + 
            blockSize * __sizeof<int>() * k + 
            blockSize * __sizeof<float>() * ns

        printfn "block size %d, shared memory size %d (%d)" blockSize sharedSize maxSharedMem

        if sharedSize > maxSharedMem then 
            failwithf "too much shared memory required: max shared mem = %d, required shared memory size = %d" maxSharedMem sharedSize

        use dC = worker.Malloc(C)
        use dDist = worker.Malloc(partitionSize) 
        use idxAndValues = worker.Malloc<IndexAndValue>(partitionSize)
        use minimum = minReductionModule.Create(partitionSize)

        let lpd = new LaunchParam(gridSize, blockSize, sharedSize)
        let lpr = LaunchParam(gridSize, blockSize)

        let findBest m =
            worker.Launch <@ distToIdentityKernelShared @> lpd m partitionSize n k ns dC.Ptr dDist.Ptr 
            worker.Launch <@ transform @> lpr dDist.Ptr idxAndValues.Ptr m partitionSize
            minimum.Reduce(idxAndValues.Ptr, partitionSize)

        let numPartitions = divup cnk (uint64 partitionSize)  
        let bestInPartitions = 
            [0UL..numPartitions - 1UL] 
            |> List.map (fun p ->  let m = p * (uint64 partitionSize) in findBest m)

        let best = bestInPartitions |> List.minBy (fun v -> v.Value)
        let bestSelection = Binomial.Cpu.subset n k best.Index
        bestSelection

    /// Find least correlated subset with all temporaries in shared memory.
    let leastCorrelated (partitionSize:int) (C : float[]) n k =
        let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT
        let maxSharedMem = worker.Device.Attributes.MAX_SHARED_MEMORY_PER_BLOCK
        let ns = size k
        let cnk = choose n k
        let blockSize = 128
        let gridSize = 8 * numSm
        let sharedSize = 
            __sizeof<float>() * size n + 
            blockSize * __sizeof<int>() * k + 
            blockSize * __sizeof<float>() * ns

        printfn "block size %d, shared memory size %d (%d)" blockSize sharedSize maxSharedMem

        if sharedSize > maxSharedMem then 
            failwithf "too much shared memory required: max shared mem = %d, required shared memory size = %d" maxSharedMem sharedSize

        use dC = worker.Malloc(C)
        use dDist = worker.Malloc<IndexAndValue>(partitionSize) 
        use minimum = minReductionModule.Create(partitionSize)

        let lpd = new LaunchParam(gridSize, blockSize, sharedSize)

        let findBest m =
            worker.Launch <@ distToIdentityKernelSharedIndexAndValue @> lpd m partitionSize n k ns dC.Ptr dDist.Ptr 
            minimum.Reduce(dDist.Ptr, partitionSize)

        let numPartitions = divup cnk (uint64 partitionSize)  
        let bestInPartitions = 
            [0UL..numPartitions - 1UL] 
            |> List.map (fun p ->  let m = p * (uint64 partitionSize) in findBest m)

        let best = bestInPartitions |> List.minBy (fun v -> v.Value)
        let bestSelection = Binomial.Cpu.subset n k best.Index
        bestSelection

    /// Using constant memory for correlation matrix. We have to build the GPU module with the cuda workflow
    /// so that we can use the GlobalArrayResource.
    let leastCorrelatedModuleUsingConstMem maxDim = cuda {
        let sizeC = size maxDim
        let! constC = Compiler.DefineConstantArray<float>(sizeC)
        let! kernel = 
            <@ fun (m0:uint64) (partitionSize:int) (n:int) (k:int) (ns:int) (dist:deviceptr<IndexAndValue>) ->
                let sizeSelection = blockDim.x * k
                let shared = __shared__.ExternArray<byte>() |> __array_to_ptr
                let selection = shared.Reinterpret<int>()
                let subMatrix = (selection + sizeSelection).Reinterpret<float>()

                let start = blockIdx.x * blockDim.x + threadIdx.x
                let stride = gridDim.x * blockDim.x

                let selection = selection + threadIdx.x * k
                let subMatrix = subMatrix + threadIdx.x * ns

                let mutable m = start  

                while m < partitionSize do
            
                    subset (fun i -> selection.[i]) (fun i v -> selection.[i] <- v) n k (m0 + uint64 m)
                    subMatixPacked (fun i -> constC.[i]) (fun i v -> subMatrix.[i] <- v) (fun i -> selection.[i]) k

                    dist.[m] <- IndexAndValue(m0 + uint64 m, distToIdentity ns (fun i -> subMatrix.[i]))

                    m <- m + stride 

            @> |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let constC = program.Apply(constC)
            let kernel = program.Apply(kernel)

            let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT
            let maxSharedMem = worker.Device.Attributes.MAX_SHARED_MEMORY_PER_BLOCK
            
            let run (partitionSize:int) (C : float[]) n k =
                if size n > sizeC then 
                    failwithf "dimension %d of C is too large only support dimension up to %d" n maxDim

                let ns = size k
                let cnk = choose n k
                let blockSize = 128
                let gridSize = 8 * numSm
                let sharedSize = 
                    blockSize * __sizeof<int>() * k + 
                    blockSize * __sizeof<float>() * ns

                printfn "block size %d, shared memory size %d (%d), const memory size %d" blockSize sharedSize maxSharedMem sizeC

                if sharedSize > maxSharedMem then 
                    failwithf "too much shared memory required: max shared mem = %d, required shared memory size = %d" maxSharedMem sharedSize
                
                constC.Scatter C
                use dDist = worker.Malloc<IndexAndValue>(partitionSize) 
                use minimum = minReductionModule.Create(partitionSize)

                let lpd = new LaunchParam(gridSize, blockSize, sharedSize)

                let findBest m =
                    kernel.Launch lpd m partitionSize n k ns dDist.Ptr 
                    minimum.Reduce(dDist.Ptr, partitionSize)

                let numPartitions = divup cnk (uint64 partitionSize)  
                let bestInPartitions = 
                    [0UL..numPartitions - 1UL] 
                    |> List.map (fun p ->  let m = p * (uint64 partitionSize) in findBest m)

                let best = bestInPartitions |> List.minBy (fun v -> v.Value)
                let bestSelection = Binomial.Cpu.subset n k best.Index
                bestSelection

            run
        )
    }

    /// Using constant memory for correlation matrix and local memory for the selection indices. 
    /// We have to build the GPU module with the cuda workflow so that we can use the GlobalArrayResource.
    let leastCorrelatedModuleUsingConstAndLocalMem maxDim maxK = cuda {
        let sizeC = size maxDim
        let! constC = Compiler.DefineConstantArray<float>(sizeC)
        let! kernel = 
            <@ fun (m0:uint64) (partitionSize:int) (n:int) (k:int) (ns:int) (dist:deviceptr<IndexAndValue>) ->
                let subMatrix = __shared__.ExternArray<float>() |> __array_to_ptr

                let start = blockIdx.x * blockDim.x + threadIdx.x
                let stride = gridDim.x * blockDim.x

                let selection = __local__.Array<int>(maxK)
                let subMatrix = subMatrix + threadIdx.x * ns

                let mutable m = start  

                while m < partitionSize do
            
                    subset (fun i -> selection.[i]) (fun i v -> selection.[i] <- v) n k (m0 + uint64 m)
                    subMatixPacked (fun i -> constC.[i]) (fun i v -> subMatrix.[i] <- v) (fun i -> selection.[i]) k

                    dist.[m] <- IndexAndValue(m0 + uint64 m, distToIdentity ns (fun i -> subMatrix.[i]))

                    m <- m + stride 

            @> |> Compiler.DefineKernel

        return Entry(fun (program:Program) ->
            let worker = program.Worker
            let constC = program.Apply(constC)
            let kernel = program.Apply(kernel)

            let numSm = worker.Device.Attributes.MULTIPROCESSOR_COUNT
            let maxSharedMem = worker.Device.Attributes.MAX_SHARED_MEMORY_PER_BLOCK
            
            let run (partitionSize:int) (C : float[]) n k =
                if size n > sizeC then 
                    failwithf "dimension %d of C is too large only support dimension up to %d" n maxDim

                let ns = size k
                let cnk = choose n k
                let blockSize = 128
                let gridSize = 8 * numSm
                let sharedSize = blockSize * __sizeof<float>() * ns

                printfn "block size %d, shared memory size %d (%d), const memory size %d" blockSize sharedSize maxSharedMem sizeC

                if sharedSize > maxSharedMem then 
                    failwithf "too much shared memory required: max shared mem = %d, required shared memory size = %d" maxSharedMem sharedSize
                
                constC.Scatter C
                use dDist = worker.Malloc<IndexAndValue>(partitionSize) 
                use minimum = minReductionModule.Create(partitionSize)

                let lpd = new LaunchParam(gridSize, blockSize, sharedSize)

                let findBest m =
                    kernel.Launch lpd m partitionSize n k ns dDist.Ptr 
                    minimum.Reduce(dDist.Ptr, partitionSize)

                let numPartitions = divup cnk (uint64 partitionSize)  
                let bestInPartitions = 
                    [0UL..numPartitions - 1UL] 
                    |> List.map (fun p ->  let m = p * (uint64 partitionSize) in findBest m)

                let best = bestInPartitions |> List.minBy (fun v -> v.Value)
                let bestSelection = Binomial.Cpu.subset n k best.Index
                bestSelection

            run
        )
    }


                
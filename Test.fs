module Test

open System
open System.Diagnostics
open NUnit.Framework
open FsUnit

open Alea.CUDA
open Alea.CUDA.Utilities
open Binomial
open LeastCorrelated

Alea.CUDA.Settings.Instance.JITCompile.Level <- "Diagnostic"
Alea.CUDA.Settings.Instance.Worker.DefaultContextType <- "threaded"

let time (f: unit -> 'A) =
    let timer = Stopwatch()
    timer.Start()
    let a = f()
    timer.Stop()
    timer.Elapsed.TotalMilliseconds, a

let timings iter (k:int list) (runner: int -> 'a) =
    k |> List.map (fun k -> 
        printfn "====> run timings for k = %d" k
        let timings = [1..iter] |> List.map (fun _ -> time (fun () -> runner k))
        let averageTime = timings |> List.map fst |> List.average
        averageTime, snd timings.[0]) 

let printTimings k f1Timings f1Text f2Timings f2Text =
    List.zip3 k f1Timings f2Timings
    |> List.map (fun (i, (t1, res1), (t2, res2)) -> 
        printfn "k = %d" i
        printfn "%s: %f ms, %A" f1Text t1 res1
        printfn "%s: %f ms, %A" f2Text t2 res2
        printfn "speedup: %f" (t1/t2))

let performanceTest iter (k:int list) (f1: int -> 'a) f1Text (f2: int -> 'a) f2Text =
    let f1Timings = timings iter k f1 
    let f2Timings = timings iter k f2 
    printTimings k f1Timings f1Text f2Timings f2Text

let randomCorr n low high = 
    let rnd = System.Random()
    let ns = size n
    Array.init ns (fun _ -> low + (high - low)*rnd.NextDouble()) |> unpack n

let C = 
    [|
        [| 1.0000; 0.9419; 0.5902; 0.6354; 0.3949; 0.2484; 0.6761; 0.7566; 0.3165; 0.0683 |]
        [| 0.9419; 1.0000; 0.0321; 0.0286; 0.3434; 0.6117; 0.2296; 0.2289; 0.7477; 0.0486 |]
        [| 0.5902; 0.0321; 1.0000; 0.2492; 0.6890; 0.5896; 0.4554; 0.7329; 0.5267; 0.4777 |]
        [| 0.6354; 0.0286; 0.2492; 1.0000; 0.7157; 0.1464; 0.6292; 0.2192; 0.4948; 0.7013 |]
        [| 0.3949; 0.3434; 0.6890; 0.7157; 1.0000; 0.1071; 0.8018; 0.8363; 0.8255; 0.8406 |]
        [| 0.2484; 0.6117; 0.5896; 0.1464; 0.1071; 1.0000; 0.8634; 0.3150; 0.2573; 0.1169 |]
        [| 0.6761; 0.2296; 0.4554; 0.6292; 0.8018; 0.8634; 1.0000; 0.1769; 0.6815; 0.5119 |]
        [| 0.7566; 0.2289; 0.7329; 0.2192; 0.8363; 0.3150; 0.1769; 1.0000; 0.6784; 0.4225 |]
        [| 0.3165; 0.7477; 0.5267; 0.4948; 0.8255; 0.2573; 0.6815; 0.6784; 1.0000; 0.0107 |]
        [| 0.0683; 0.0486; 0.4777; 0.7013; 0.8406; 0.1169; 0.5119; 0.4225; 0.0107; 1.0000 |]
    |]

[<Test>]
let ``Packing lower triangular matrix is correct`` () =
    let A' = lowerTriangularPacked C
    let A1 = unpack 10 A'

    A1 |> should equal C

[<Test>]
let ``All combinations found`` () =

    let n = choose 10 3 |> int

    let all = List.init n (fun i -> Cpu.subset 10 3 (uint64 i))

    let allDistinct = all |> Seq.distinct |> Seq.toList
    
    List.length allDistinct |> should equal n

[<Test>]
let ``Pack and unpack correlation matrix is correct`` () =
    let C' = lowerTriangularPacked C
    let C1 = unpack 10 C'
    C1 |> should equal C

[<Test>]
let ``Pack and unpack correlation submatrix is correct`` () =
    let C' = lowerTriangularPacked C
    let sel = [|0; 1; 4; 9|]
    let Asub' = Cpu.subMatixPacked C' sel
    let Asub = unpack 4 Asub'

    let expected = 
        [|
            [| 1.0000; 0.9419; 0.3949; 0.0683 |]
            [| 0.9419; 1.0000; 0.3434; 0.0486 |]
            [| 0.3949; 0.3434; 1.0000; 0.8406 |]
            [| 0.0683; 0.0486; 0.8406; 1.0000 |]
        |]

    Asub |> should equal expected

[<Test>]
let ``Cpu verions of least correlated set is correct`` () =
    let C' = lowerTriangularPacked C
    let bestSelection = Cpu.leastCorrelated C' 10 5
    let best = Cpu.subMatixPacked C' bestSelection |> unpack 5

    bestSelection |> should equal [|1; 3; 5; 7; 9|]
    
    let expected = 
        [|
            [| 1.0000; 0.0286; 0.6117; 0.2289; 0.0486 |]
            [| 0.0286; 1.0000; 0.1464; 0.2192; 0.7013 |]
            [| 0.6117; 0.1464; 1.0000; 0.3150; 0.1169 |]
            [| 0.2289; 0.2192; 0.3150; 1.0000; 0.4225 |]
            [| 0.0486; 0.7013; 0.1169; 0.4225; 1.0000 |]        
        |]
    
    best |> should equal expected

let testDistToIdentity k =
    let n = C.Length  
    let C' = lowerTriangularPacked C
    let dist = Gpu.allDistToIdentityDevice C' n k
    let dist' = Cpu.allDistToIdentity C' n k
    Array.zip dist dist' 
    |> Array.iter (fun (d, (_, d')) -> let err = abs (d - d') in err |> should (equalWithin 1e-10) 0.0)

[<Test>]
let ``Cpu and Gpu calculate same distance to identity for k = 3`` () = testDistToIdentity 3

[<Test>]
let ``Cpu and Gpu calculate same distance to identity for k = 4`` () = testDistToIdentity 4

[<Test>]
let ``Cpu and Gpu calculate same distance to identity for k = 5`` () = testDistToIdentity 5

let cpu C n k =
    printfn "cpu n = %d, k = %d" n k
    Cpu.leastCorrelated C n k

let gpu C n k =
    let partitionSize = choose n k |> int
    printfn "gpu n = %d, k = %d" n k
    Gpu.leastCorrelated partitionSize C n k

[<Test>]
let ``Performance benchmark - Cpu vs Gpu shared memory n = 20, k = [5..9]`` () =
    let C = randomCorr 20 0.0 0.95 |> lowerTriangularPacked   
    let n = 20
    let k = [5..9]
    gpu C n k.[0] |> ignore  
    performanceTest 5 k (cpu C n) "Cpu" (gpu C n) "Gpu" |> ignore

[<Test>]
let ``Performance benchmark - Cpu vs Gpu shared memory n = 50, k = [3..5]`` () =
    let C = randomCorr 50 0.0 0.95 |> lowerTriangularPacked        
    let n = 50
    let k = [3..5]
    gpu C n k.[0] |> ignore  
    performanceTest 5 k (cpu C n) "Cpu" (gpu C n) "Gpu" |> ignore

let compileLeastCorrelatedConstMem maxDim =
    let gpuModule = Gpu.leastCorrelatedModuleUsingConstMem maxDim
    let program = gpuModule |> Compiler.load Worker.Default
    program.Run

[<Test>]
let ``Constant memory version of least correlated set agree`` () =
    let leastCorrelated = compileLeastCorrelatedConstMem 50
    let C' = lowerTriangularPacked C
    let n = C.Length
    let k = 4
    let partitionSize = choose n k |> int
    let gpuShared = Gpu.leastCorrelated partitionSize C' n k
    let gpuConst = leastCorrelated partitionSize C' n k

    printfn "gpu const mem: %A, gpu shared mem %A" gpuConst gpuShared
    
    gpuShared |> should equal gpuShared

let compileLeastCorrelatedConstAndLocalMem maxDim maxK =
    let gpuModule = Gpu.leastCorrelatedModuleUsingConstAndLocalMem maxDim maxK
    let program = gpuModule |> Compiler.load Worker.Default
    program.Run

[<Test>]
let ``Constant and local memory version of least correlated set agree`` () =
    let leastCorrelated = compileLeastCorrelatedConstAndLocalMem 50 4
    let C' = lowerTriangularPacked C
    let n = C.Length
    let k = 4
    let partitionSize = choose n k |> int
    let gpuShared = Gpu.leastCorrelated partitionSize C' n k
    let gpuConstAndLocal = leastCorrelated partitionSize C' n k

    printfn "gpu const and local mem: %A, gpu shared mem %A" gpuConstAndLocal gpuShared
    
    gpuShared |> should equal gpuShared

let testPerformanceConstMem n k = 
    let leastCorrelated = compileLeastCorrelatedConstMem 50
    let C = randomCorr 50 0.0 0.95 |> lowerTriangularPacked        
    let partitionSize k = choose n k |> int
    let gpuShared k = Gpu.leastCorrelated (partitionSize k) C n k 
    let gpuConst k = leastCorrelated (partitionSize k) C n k
    gpuShared |> ignore 
    gpuConst |> ignore
    performanceTest 5 k gpuShared "Shared mem" gpuConst "Const shared mem" |> ignore

[<Test>]
let ``Performance benchmark - Gpu constant shared memory vs Gpu shared memory n = 20, k = [5..9]`` () =
    testPerformanceConstMem 20 [5..9]

[<Test>]
let ``Performance benchmark - Gpu constant shared memory vs Gpu shared memory n = 50, k = [3..5]`` () =
    testPerformanceConstMem 50 [3..5]

let testPerformanceConstAndLocalMem n k =
    let leastCorrelated = compileLeastCorrelatedConstAndLocalMem 50 10
    let C = randomCorr 50 0.0 0.95 |> lowerTriangularPacked        
    let partitionSize k = choose n k |> int
    let gpuShared k = Gpu.leastCorrelated (partitionSize k) C n k 
    let gpuConst k = leastCorrelated (partitionSize k) C n k
    gpuShared |> ignore 
    gpuConst |> ignore
    performanceTest 5 k gpuShared "Shared mem" gpuConst "Const local shared mem" |> ignore

[<Test>]
let ``Performance benchmark - Gpu constant local shared memory vs Gpu shared memory n = 20, k = [5..9] `` () =
    testPerformanceConstAndLocalMem 20 [5..9]

[<Test>]
let ``Performance benchmark - Gpu constant local shared memory vs Gpu shared memory n = 50, k = [3..5] `` () =
    testPerformanceConstAndLocalMem 50 [3..5]    

let testPerformanceCpuGpuConstAndLocalMem n (k:int list) =
    let leastCorrelated = compileLeastCorrelatedConstAndLocalMem 50 10
    let C = randomCorr 50 0.0 0.95 |> lowerTriangularPacked        
    let partitionSize k = choose n k |> int
    let gpu k = leastCorrelated (partitionSize k) C n k        
    gpu k.[0] |> ignore  
    performanceTest 5 k (cpu C n) "Cpu" gpu "Gpu const local shared" |> ignore
    
[<Test>]
let ``Performance benchmark - Cpu vs Gpu constant local shared memory n = 20, k = [5..10]`` () =
    testPerformanceCpuGpuConstAndLocalMem 20 [5..10] 

[<Test>]
let ``Performance benchmark - Cpu vs Gpu constant local shared memory n = 50, k = [3..5]`` () =
    testPerformanceCpuGpuConstAndLocalMem 50 [3..5] 
    

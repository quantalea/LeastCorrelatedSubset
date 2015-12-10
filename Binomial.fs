module Binomial

[<ReflectedDefinition>]
let choose (n:int) (k:int) =
    if k > n then 
        0UL
    else if k = 0 || k = n then 
        1UL
    else if k = 1 || k = n - 1 then
        uint64 n
    else 
        let delta, iMax = if k < n - k then uint64 (n - k), k else uint64 k, n - k
        let mutable res = delta + 1UL
        for i = 2 to iMax do
            res <- (res * (delta + uint64 i)) / (uint64 i)
        res

[<ReflectedDefinition>]
let largest a b x =
    let mutable v = a - 1
    while choose v b > x do
        v <- v - 1
    v

[<ReflectedDefinition>]
let subset (getSub : int -> int) (setSub : int -> int -> unit) (n:int) (k:int) (m:uint64) =   
    let mutable x = (choose n k) - 1UL - m
    let mutable a = n
    let mutable b = k

    for i = 0 to k-1 do
        largest a b x |> setSub i
        x <- x - choose (getSub i) b
        a <- getSub i
        b <- b - 1

    for i = 0 to k-1 do
        (n - 1) - getSub i |> setSub i

module Cpu =

    let subset (n:int) (k:int) (m:uint64) =
        let sub = Array.zeroCreate k
        subset (fun i -> sub.[i]) (fun i v -> sub.[i] <- v) n k m
        sub


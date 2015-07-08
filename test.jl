areload()

srand(1)

data = matread("/home/90days/ko/ml-latest-small/ml-latest-100k.mat")
I = convert(Array{Int64,1},vec(data["I"]))
J = convert(Array{Int64,1}, vec(data["J"]))
V = vec(data["V"])
M = maximum(I)
N = maximum(J)

Y = sparse(I,J,V, M,N)


Ytrain,Ytest = splitData(Y, 0.25)

keep = find(sum(Ytrain , 2) .> 0)
Ytrain = Ytrain[keep,:]
Ytest = Ytest[keep,:]
M,N = size(Ytrain);
matwrite("/tmp/test.mat", {"Ytrain" => Ytrain, "Ytest" => Ytest,"Y"=> Y})

options = ALSOptions(50)

#Ds =[5,10,25,50,100]
Ds =[25]
TsALS =zeros(length(Ds))
TsCCD =zeros(length(Ds))

objcpp = []
objals = []

outmat = Dict()

println("Running")
for (d, D) in enumerate(Ds)
    params =  Parameter(D,15.0,15.0)
    V,U, objals, errals, errtrals, tals = @time als(Ytrain,Ytest, params,options)
    V,U, objccd, errccd, errtrccd, tccd = @time ccdpp(Ytrain,Ytest, params,options)
    TsALS[d] = mean(tals)
    TsCCD[d] = mean(tccd)
    outmat["tals"] = tals
    outmat["objals"] = objals
    outmat["errals"] = errals
    outmat["errtrals"] = errtrals

    outmat["tccd"] = tccd
    outmat["objccd"] = objccd
    outmat["errccd"] = errccd
    outmat["errtrccd"] = errtrccd
end

matwrite("/tmp/myout2.mat", outmat)
#matwrite("/tmp/myout.mat", {"Tals"=>TsALS,"Tccd"=>TsCCD})

#matwrite("/tmp/myout.mat", {"V"=>V,"U"=>U})
#@profile  ccdpp(Ytrain,Ytest, params,options)

#matwrite("als.mat", {"objectives" => objectives, "errtrain" => errtrain, "errtest" => errtest, "D"=> params.D, "lambdaU"=>params.lambdaU, "lambdaV" => params.lambdaV})



areload()

srand(1)

data = matread("/home/90days/ko/ml-latest-small/ml-latest-100k.mat")
I = convert(Array{Int64,1},vec(data["I"]))
J = convert(Array{Int64,1}, vec(data["J"]))
V = vec(data["V"])
M = maximum(I)
N = maximum(J)

Y = sparse(I,J,V, M,N)
params =  Parameter(50,5.0,5.0)

options = ALSOptions(20)

Ytrain,Ytest = splitData(Y, 0.25)

keep = find(sum(Ytrain , 2) .> 0)
Ytrain = Ytrain[keep,:]
Ytest = Ytest[keep,:]
M,N = size(Ytrain);
#options.V0 = rand(M,D) *0.1
matwrite("/tmp/test.mat", {"Ytrain" => Ytrain, "Ytest" => Ytest,"Y"=> Y})
mu = mean(Ytrain)




println("bla")
@time objectives, errtrain, errtest = als(Ytrain,Ytest, params,options)
V,U = @time ccdpp(Ytrain,Ytest, params,options)
matwrite("/tmp/myout.mat", {"V"=>V,"U"=>U})
#@profile  ccdpp(Ytrain,Ytest, params,options)

#matwrite("als.mat", {"objectives" => objectives, "errtrain" => errtrain, "errtest" => errtest, "D"=> params.D, "lambdaU"=>params.lambdaU, "lambdaV" => params.lambdaV})



function updateU!(Ytrain, U,V, bu, bi, avg, lambda)

    M,N = size(Ytrain)
    D = size(U, 1)

    for n in 1:N
        #nObs = getNObsCol(Ytrain, n)

        if Ytrain.colptr[n+1] == Ytrain.colptr[n]
            continue
        end

        obsIdx, yobs = getObsCol(Ytrain, n)
        Vobs = V[:,obsIdx]

        U[:,n] = ( Vobs * Vobs' + lambda * eye(D)) \ (Vobs * (yobs - bi[obsIdx] - bu[n] - avg))

    end
end

function updateBias!(Ytrain, U,V, bu, bi, avg, lambda)

    M,N = size(Ytrain)
    D = size(U, 1)

    for n in 1:N
        nObs::Float64 = getNObsCol(Ytrain, n)

        if nObs == 0 
            continue
        end

        obsIdx, yobs = getObsCol(Ytrain, n)
        bu[n] = sum(yobs - V[:,obsIdx]' * U[:,n] - bi[obsIdx] - avg) / (nObs + lambda)

    end


end

function sumSquaredRes(I,J,R, V, U, bu, bi, mu)
    D = size(V,1)
    error = 0.0
    for (ridx,r) in enumerate(R)
        dprod = 0.0
        for it =1:D
            dprod += V[it, I[ridx]] * U[it, J[ridx]]
        end
        res = r - dprod - mu - bu[J[ridx]] - bi[I[ridx]]
        error += res * res
    end
    return error   
end

function als(Ytrain, Ytest, params, options)

    M,N = size(Ytrain)
    D = params.D

    srand(1)
    V = rand(D,M) * 0.1
    U = rand(D,N) * 0.1

    mu = mean(Ytrain.nzval)
    bu = zeros(N)
    bi = zeros(M)

    Ite,Jte,Vte = findnz(Ytest)
    Itr,Jtr,Vtr = findnz(Ytrain)

    YtrainT = Ytrain'

    objectives = zeros(options.maxIt)
    errtrain = zeros(options.maxIt)
    errtest = zeros(options.maxIt)
    timing = zeros(options.maxIt)

    for it in 1:options.maxIt
        tic()
        updateU!(Ytrain, U,V, bu,bi, mu, params.lambdaU)
        updateU!(YtrainT, V,U, bi,bu, mu, params.lambdaV)
        timing[it] = toq()

        #updateBias!(Ytrain, U,V, bu,bi, mu, params.lambdaU)
        #updateBias!(YtrainT, V,U, bi,bu, mu, params.lambdaV)

        #Test Error
        ssq = sumSquaredRes(Ite,Jte,Vte, V, U, bu, bi, mu)
        errtest[it] =sqrt(ssq/length(Vte))

        #Objective function
        obj = sumSquaredRes(Itr,Jtr,Vtr, V, U, bu, bi, mu)
        errtrain[it] = sqrt(obj/length(Vtr))
        #add regularization terms
        obj += params.lambdaV * dot(V[:],V[:]) 
        obj += params.lambdaU * dot(U[:],U[:]) 
        obj += params.lambdaV * dot(bi,bi)
        obj += params.lambdaU * dot(bu,bu) 

        objectives[it] = obj

        println(it, "\t RMSE:\t", errtest[it] , " \t Objective:\t", obj)
    end
    return V,U, objectives, errtest, errtrain,timing
end


function updateU1!(Ytrain, U,V, avg, lambda)

    M,N = size(Ytrain)
    D = size(U, 1)

    for n in 1:N
        @inbounds if Ytrain.colptr[n+1] == Ytrain.colptr[n]
            continue
        end

        num = 0.0
        den = lambda

        @inbounds for o in ( Ytrain.colptr[n]  : Ytrain.colptr[n+1]-1)
            @inbounds vo = V[Ytrain.rowval[o]]
            @inbounds num += (Ytrain.nzval[o] - avg) * vo 
            den += vo * vo
        end

        # CCD update: dot(yobs - mu, vobs) / (lambda + dot(vobs, vobs))
        U[n] = num / den

    end
end



function solveRank1!(Rhat,RhatT,mu, u,v, params,options)

    for it in 1:options.maxIt
        updateU1!(Rhat, u,v, mu, params.lambdaU)
        updateU1!(RhatT, v,u, mu, params.lambdaV)
    end

end

function updateR!(R, u,v, add)

    if add
        for n in 1:length(u) 
            nObs = getNObsCol(R, n)
            if nObs == 0
                continue
            end
            for c in (R.colptr[n]: R.colptr[n+1] -1)
                R.nzval[c] += v[R.rowval[c]] * u[n] 
            end
        end
    else
        for n in 1:length(u) 
            nObs = getNObsCol(R, n)
            if nObs == 0
                continue
            end
            for c in (R.colptr[n]: R.colptr[n+1] -1)
                R.nzval[c] -= v[R.rowval[c]] * u[n] 
            end
        end
    end
end

function ccdpp(Ytrain, Ytest, params, options)
    srand(100)
    D = params.D
    M,N = size(Ytrain)
    U = zeros(N,D)
    V = rand(M,D) * 0.1
    #V = ones(M,D) * 0.1
    #    V = options.V0;
    R = copy(Ytrain) # due to init U=0

    mu = mean(Ytrain.nzval)
    bu = zeros(N)
    bi = zeros(M)
    #R.nzval = R.nzval - mu
    RT =  R'

    I,J,Vals = findnz(Ytrain)

    Ite,Jte,ValsTe = findnz(Ytest)
    #ValsTe -= mu
    mm = mu
    #mu = 0.0 

    optionsInner = ALSOptions(1) #number of inner iterations

    bu = zeros(N)
    bi = zeros(M)
    timing = zeros(options.maxIt)
    errtest = zeros(options.maxIt)
    errtrain = zeros(options.maxIt)
    objectives = zeros(options.maxIt)

    nObs = convert(Float64, length(Vals))
    for it in 1:options.maxIt
        tic()

        for d in 1:D
            u = unsafe_view(U, :, d)
            v = unsafe_view(V, :, d)

            updateR!(R, u,v, true)
            updateR!(RT, v,u, true)

            solveRank1!(R,RT,mu, u,v, params,optionsInner)

            updateR!(R, u,v, false)
            updateR!(RT, v,u, false)
        end
        timing[it] = toq()

        ssq = sumSquaredRes(Ite,Jte,ValsTe, V', U', bu, bi, mu)
        err =sqrt(ssq/length(ValsTe))
        errtest[it] = err
        #println("Iter: ", it, " RMSE: ", err )

        #Objective function
        obj = sumSquaredRes(I,J,Vals, V', U', bu, bi, mm)

        errtrain[it] = sqrt(obj/nObs)
        #add regularization terms
        obj += params.lambdaV * dot(V[:],V[:]) 
        obj += params.lambdaU * dot(U[:],U[:]) 
        obj += params.lambdaV * dot(bi,bi)
        obj += params.lambdaU * dot(bu,bu) 

        objectives[it] = obj

        println(it, "\t Tr.Err $(errtrain[it])\t RMSE:\t", errtest[it] , " \t Objective:\t", obj)

    end
    return V,U, objectives, errtest, errtrain,timing
end


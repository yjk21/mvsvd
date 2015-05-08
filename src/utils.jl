function rmse(Ytest::SparseMatrixCSC, pred::Array{Float64})
    res = Ytest.nzval - pred;
    return sqrt( dot(res,res) / length(pred))
end


function getNObsCol(Y::SparseMatrixCSC, n::Int64)
    return Y.colptr[n+1] - Y.colptr[n]
end

function getObsCol(Y::SparseMatrixCSC, n::Int64)
    obsIdx = unsafe_view(Y.rowval, Y.colptr[n]  : Y.colptr[n+1]-1)
    yobs = unsafe_view(Y.nzval, Y.colptr[n] : Y.colptr[n+1]-1)
    return obsIdx, yobs
end

function splitData(Y, p)
    # Split data into train and test. From each user we take a fraction p of the observations  
    II = Int64[]
    JJ = Int64[]
    VV = Float64[]

    IIte = Int64[]
    JJte = Int64[]
    VVte = Float64[]

    M,N = size(Y)

    for n = 1:N
        nObs = getNObsCol(Y,n)
        if nObs == 0
            continue
        end
        obsIdx, = getObsCol(Y,n)

        pIdx = randperm(nObs)

        nTest = (floor(p * nObs))
        nTrain = nObs - nTest

        for i in 1:nObs 
            idx = obsIdx[pIdx[i]]
            if i <= nTest
                push!(IIte, idx)
                push!(JJte, n)
                push!(VVte, Y[idx,n])
            else
                push!(II, idx)
                push!(JJ, n)
                push!(VV, Y[idx,n])
            end
        end
    end

    Ytrain = sparse(II,JJ,VV, M,N)
    Ytest = sparse(IIte,JJte,VVte,M,N)
    return Ytrain, Ytest
end



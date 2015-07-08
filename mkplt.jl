using MAT
using PyPlot
close("all")
figure(figsize=(6,7))
raw = matread("/tmp/myout.mat")
Tals = raw["Tals"]
Tccd = raw["Tccd"]
Ds =[5,10,25,50,100]

plot(Ds, Tals, marker="x", color="red", markersize=10, markeredgewidth=3)
plot(Ds, Tccd, marker="o", color="blue", markersize=10, markeredgewidth=3)
xlabel("Latent Factors D")
ylabel("Time per Iteration (sec.)")
xlim([2, 102])
grid("on")
legend(["ALS","CCD"])
title("Scalability wrt. Latent Dimensionality")
savefig("/home/90days/ko/exp.pdf", format="pdf")

figure(figsize=(6,7))
raw2 = matread("/tmp/myout2.mat")
tals =raw2["tals"] 
objals = raw2["objals"] 
errals =raw2["errals"] 
erertrals = raw2["errtrals"] 

tccd = raw2["tccd"] 
objccd = raw2["objccd"] 
errccd = raw2["errccd"] 
errtrccd =raw2["errtrccd"] 

plot(cumsum(tals), objals,color="red")
plot(cumsum(tccd), objccd,color="blue")

title("Convergence (D=10)")
xlabel("Time (sec.)")
ylabel("Objective Function Value")

xlim([0,6])
grid("on")
legend(["ALS","CCD"])
savefig("/home/90days/ko/obj.pdf", format="pdf")

figure(figsize=(6,7))

plot(cumsum(tals), errals,color="red")
plot(cumsum(tccd), errccd,color="blue")

title("Test Error (D=10)")
xlabel("Time (sec.)")
ylabel("RMSE")
xlim([0,6])
ylim([0.94,1.02])
grid("on")

legend(["ALS","CCD"])
savefig("/home/90days/ko/err.pdf", format="pdf")




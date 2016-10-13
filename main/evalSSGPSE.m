function [trainingTime,testTime,hyp,nlml,mYest,vYest] = ...
    evalSSGPSE(Xtrain,Ytrain,Xtest,M,params,noEvals)

[Ntrain,D] = size(Xtrain);

lengthscale = params(1);
covpower = params(2);
noisepower = params(3);

tic
hypers = [lengthscale;covpower;noisepower];
spectralPoints = randn(M*D,1);
% training
params=[hypers; spectralPoints];
iterLength = -noEvals;
[params,fi,i] = minimize(params,'ssgpr',iterLength,Xtrain,Ytrain);
hyp.cov = params(1:D+1);
hyp.lik = params(D+2);
hyp.spectralPoints = params(D+3:end);
nlml = fi(end);
trainingTime = toc;
% prediction
tic
[mYest, vYest] = ssgpr(params, Xtrain, Ytrain, Xtest);
testTime = toc;
end



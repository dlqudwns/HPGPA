function [trainingTime,testTime,hyp,nlml,my,vy] = ...
    evalVarSE(Xtrain,Ytrain,Xtest,M,logtheta0,covfunc,noEvals)

[Ntrain,D] = size(Xtrain);
z = Xtrain(randperm(Ntrain,M),:);
% training
tic
[theta, nlml] = vfeTrain(logtheta0,covfunc,Xtrain,Ytrain,noEvals,z);
trainingTime = toc;
noCovParams = eval(feval(covfunc{:}));
hyp.cov = theta(1:noCovParams);
hyp.lik = theta(noCovParams+1);
hyp.Xu = z;
% prediction
tic
[my,vf] = vfePredict(covfunc,hyp,Xtrain,Ytrain,Xtest);
testTime = toc;
% add noise variance sigma2 to get a prediction for the ys
vy = vf + exp(2*hyp.lik);
end

function [trainingTime,testTime,hyp,nlml,my,vy] = ...
    evalSSMSE(Xtrain,Ytrain,Xtest,approxDeg,logtheta0,noEvals)
% training
tic
[theta_end,nlml] = trainSSM_SE(logtheta0,Xtrain,Ytrain,approxDeg,noEvals);
trainingTime = toc;
hyp.cov = theta_end(1:2);
hyp.lik = theta_end(3);
% prediction
tic
[my,vf] = predictSSM_SE(theta_end,Xtrain,Ytrain,Xtest,approxDeg);
testTime = toc;
% add noise variance sigma2 to get a prediction for the ys
vy = vf + exp(2*hyp.lik);
end

function [fest,vest] = predictSSM_SE(theta,Xtrain,Ytrain,Xtest,approxDeg)
% init params
ell = exp(theta(2));
sig2 = exp(theta(1));
sn2 = exp(theta(3));
% likelihood
likfunc = lik_gaussian('sigma2',sn2);
% covariance function
covfunc = gpcf_sexp('lengthscale',ell,'magnSigma2',sig2,'kalman_deg',approxDeg);
gpHandle = gp_set('lik',likfunc,'cf',covfunc,'type','KALMAN');
[fest,vest] = gp_pred(gpHandle,Xtrain,Ytrain,'xt',Xtest);
end

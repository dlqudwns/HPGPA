function [theta_end,nlml] = trainSSM_SE(theta_init,Xtrain,Ytrain,approxDeg,noEvals)

% TODO: test this!

% init params
ell = exp(theta_init(1));
sig2 = exp(2*theta_init(2));
sn2 = exp(2*theta_init(3));

% likelihood
likfunc = lik_gaussian('sigma2',sn2);
% covariance function
covfunc = gpcf_sexp('lengthscale',ell,'magnSigma2',sig2,'kalman_deg',approxDeg);
gpHandle = gp_set('lik',likfunc,'cf',covfunc,'type','KALMAN');

% instead of gpstuff optimisation, we use Carl's minimize function for a fair
% comparison
w_init=gp_pak(gpHandle);
% d = checkgrad('gp_eg',w_init,1e-5,gpHandle,Xtrain,Ytrain)
% keyboard
[w_end,vals] = minimize(w_init,'gp_eg',-noEvals,gpHandle,Xtrain,Ytrain);

theta_end = w_end;
nlml = vals(end);
end
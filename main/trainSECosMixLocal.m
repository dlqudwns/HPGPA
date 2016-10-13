function [theta_end,nlml] = trainSECosMixLocal(theta_init,K,covfunc,...
    Xtrain,Ytrain,tau1,tau2,missingInd,noEvals)


fname = 'objFunctionSECosMixLocal';

% %test derivative
% d = checkgrad(fname,theta_init,1e-6,K,covfunc,...
%     Xtrain,yKalman,tau1,tau2,missingInd)
% keyboard

opt.length = -noEvals;
opt.method = 'BFGS';
opt.verbosity = 2;
[theta_end,lik,i] = minimize_new(theta_init,fname,opt,K,covfunc,...
    Xtrain,Ytrain,tau1,tau2,missingInd);
nlml = lik(end);
end
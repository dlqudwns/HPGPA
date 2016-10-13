function [theta_end,nlml] = trainSEHier_single(theta_init,covfunc,...
    Xtrain,Ytrain,tau1,tau2,tau3,missingInd,noEvals)

Ntrain = length(Xtrain);
noblks = Ntrain/tau1;
y1 = reshape(Ytrain,tau1,noblks);
yKalman = num2cell(y1,1);
fname = 'objFunctionSEHier_single';
% test derivativevest(ind) = diag(C1*Pfint{i}*C1' + R1);
% checkgrad('gPSHwrapper',theta_init(1),1e-4,theta_init,covfunc,...
%      Xtrain,tau1,tau2,tau3,noblks,missingInd)
%  keyboard

%checkgrad('objFunctionSEHier_single', theta_init, 1e-4, covfunc, Xtrain, yKalman, tau1, tau2, tau3, missingInd);
% checkgrad('objFunctionSE', theta_init, 1e-4, covfunc, Xtrain, yKalman, tau1, tau2, missingInd);
%keyboard
opt.length = -noEvals;
opt.method = 'BFGS';
opt.verbosity = 0;
[theta_end,lik,i] = minimize_new(theta_init,fname,opt,covfunc,...
    Xtrain,yKalman,tau1,tau2,tau3,missingInd);
nlml = lik(end);
end
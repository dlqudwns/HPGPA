function [theta_end,nlml] = trainSEHier(theta_init,covfunc,...
    cross_covfunc,Xtrain,Ytrain,tau1,tau2,tau3,missingInd,noEvals,H)

K = length(Xtrain)/tau1;
if size(Xtrain, 2) == 1 % 1d
    y1 = reshape(Ytrain,tau1,K);
    yKalman = num2cell(y1,1);    
elseif size(Xtrain, 2) == 2 % 2d
    yKalman = {};
    for j=1:K
        for i=1:K
            yKalman{end+1} = Ytrain((i-1)*tau1+1:i*tau1, (j-1)*tau1+1:j*tau1);
            yKalman{end} = yKalman{end}(:);
        end
    end
else
    error('dimension of Xtrain?')
end
fname = 'objFunctionSEHier';
% test derivativevest(ind) = diag(C1*Pfint{i}*C1' + R1);

% fprintf('Check grad gPSHwrapper\n');
% for j=1:length(theta_init)
%     checkgrad('gPSHwrapper',theta_init(j),1e-4,j,theta_init,covfunc,cross_covfunc,...
%         Xtrain,tau1,tau2,tau3,noblks,missingInd);
% end
% fprintf('Check grad objFunctionSEHier\n');
% checkgrad('objFunctionSEHier', theta_init, 1e-2, covfunc,cross_covfunc, Xtrain, yKalman, tau1, tau2, tau3, missingInd, H);
% fprintf('Check grad end!\n');
% keyboard

%checkgrad('objFunctionSE', theta_init, 1e-4, covfunc, Xtrain, yKalman, tau1, tau2, missingInd);
%hyp1 = theta_init(1:2);
%hyp2 = theta_init(3:4);
%a = covSEiso(hyp1, Xtrain(1:10));
%b = covSEiso(hyp2, Xtrain(1:10));
%ab = crosscovSEiso(hyp1, hyp2, Xtrain(1:10));
%E = [a,ab;ab',b]
%chol(E);
%checkgrad('covFuncchecker', theta_init(1:4), 1e-4, Xtrain(1:100))

opt.length = -noEvals;
opt.method = 'BFGS';
opt.verbosity = 0;
[theta_end,lik,i] = minimize_new(theta_init,fname,opt,covfunc,...
    cross_covfunc,Xtrain,yKalman,tau1,tau2,tau3,missingInd,H);
nlml = lik(end);
end
function [theta_end,nlml] = trainSEFITC(theta_init,covfunc,...
    Xtrain,Ytrain,M,noEvals)

if size(Xtrain, 2) == 1 % 1d
    inducing = linspace(min(Xtrain),max(Xtrain),M);
elseif size(Xtrain, 2) == 2 % 2d
    inducing = splitSpace(Xtrain(1,:), Xtrain(end,:), M);
end
likFunc = @likGauss;
covFunc = {@covFITC,covfunc,inducing'};
hyp.cov = theta_init(1:end-1); 
hyp.lik = theta_init(end);

[theta_end,nlml] = minimize(hyp,@gp,-noEvals,@infFITC,[],covFunc, ...
    likFunc,Xtrain,Ytrain);
end
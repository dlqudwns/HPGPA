function [theta_end,nlml] = trainSECosFITC(theta_init,covfunc,...
    Xtrain,Ytrain,M,noEvals)

inducing = linspace(min(Xtrain),max(Xtrain),M);
likFunc = @likGauss;
covFunc = {@covFITC,covfunc,inducing'};
hyp.cov = theta_init(1:end-1); 
hyp.lik = theta_init(end);

[theta_end,nlml] = minimize(hyp,@gp,-noEvals,@infFITC,[],covFunc, ...
    likFunc,Xtrain,Ytrain);
end
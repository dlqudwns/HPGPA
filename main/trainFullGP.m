function [theta_end, nlml] = trainFullGP(theta_init,covfunc,Xtrain,Ytrain,noEvals)

likFunc = @likGauss;
hyp.cov = theta_init(1:end-1); 
hyp.lik = theta_init(end);

[theta_end,nlml] = minimize(hyp,@gp,-noEvals,@infExact,[],covfunc,likFunc,Xtrain,Ytrain);

end
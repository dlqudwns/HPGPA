function [theta_end,nlml] = trainVSGP(theta_init,...
    Xtrain,Ytrain,M,noEvals)

if size(Xtrain, 2) == 1 % 1d
    inducing = linspace(min(Xtrain),max(Xtrain),M);
elseif size(Xtrain, 2) == 2 % 2d
    inducing = splitSpace(Xtrain(1,:), Xtrain(end,:), M);
end

lengthscales = ones(M,1) * theta_init(1);

likFunc = @likGauss;
covFunc = {@covVSGP,inducing'};
hyp.cov = [theta_init(1:end-1); lengthscales];
hyp.lik = theta_init(end);

[theta_end,nlml] = minimize(hyp,@gp,-noEvals,@infVSGP,[],covFunc, ...
    likFunc,Xtrain,Ytrain);
end
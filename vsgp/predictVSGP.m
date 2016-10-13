function [fest,vest] = predictVSGP(theta,Xtrain,Ytrain,Xtest,M)

if size(Xtrain,2) == 1 %1d
    inducing = linspace(min(Xtrain),max(Xtrain),M);
elseif size(Xtrain,2) == 2 %2d
    inducing = splitSpace(Xtrain(1,:), Xtrain(end,:), M);
end

likFunc = @likGauss;
hyp = theta;
covFunc = {@covVSGP,inducing'};
[~,~,fest,vest] = gp(hyp,@infVSGP,[],covFunc,likFunc,Xtrain,Ytrain,Xtest);
end


function [fest,vest] = predictFullGP(theta,covfunc,Xtrain,Ytrain,Xtest)

likFunc = @likGauss;
hyp = theta;
[~,~,fest,vest] = gp(hyp,@infExact,[],covfunc,likFunc,Xtrain,Ytrain,Xtest);

end


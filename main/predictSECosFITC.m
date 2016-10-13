function [fest,vest] = predictSECosFITC(theta,covfunc,Xtrain,Ytrain,Xtest,M)

inducing = linspace(min(Xtrain),max(Xtrain),M);
likFunc = @likGauss;
hyp = theta;
covFunc = {@covFITC,covfunc,inducing'};
[~,~,fest,vest] = gp(hyp,@infFITC,[],covFunc,likFunc,Xtrain,Ytrain,Xtest);
end


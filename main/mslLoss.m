function loss = mslLoss(fTrue,fEst,varEst,meanTrain,varTrain)
if isempty(fTrue) || isempty(fEst) || isempty(varEst) || ...
        length(fTrue)~=length(fEst)
    loss = inf;
    error('Empty input vector or vectors of different length')
    %return;
end

err = 0.5*mean(((fTrue-fEst)./sqrt(varEst)).^2+log(varEst));

err1 = 0.5*mean(((fTrue-meanTrain)/sqrt(varTrain)).^2+ ...
                log(varTrain));
loss = err - err1;
end

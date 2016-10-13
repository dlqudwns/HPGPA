function [f,df] = objFunctionSEHier_single(theta,covfunc,X,Y,tau1,tau2,tau3,missingInd)


%% get model parameters and the derivatives
noblks = length(Y);
vary = exp(theta(end));
[Ct,Rt,At,Qt,pIndex,dCt,dRt,dAt,dQt] = ...
    getParamsSEHier_single(covfunc,theta(1:end-1),vary,X,tau1,tau2,tau3,noblks,missingInd);

% initial condition
u1 = splitSpace(X(1),X(end),tau2)';
alpha = eval(feval(covfunc{:}));
Kg = feval(covfunc{:},theta(1:alpha),u1) + 1e-7*eye(tau2);
dKg = cell(alpha+1,1);
for i = 1:alpha
    dP0i = feval(covfunc{:},theta(1:alpha),u1,[],i);
    dKg{i} = dP0i;
end
dKg{end} = zeros(size(Kg));

[f,df] = kalmanVarHier_single(At,Ct,Qt,Rt,pIndex,Kg,Y,...
    dAt,dCt,dQt,dRt,dKg);
f = -f/(noblks*tau1);
df = -df/(noblks*tau1);
end


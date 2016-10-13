function [fest,vest] = predictSEHier_single(theta,covfunc,Xtrain,Ytest,tau1,tau2,tau3,missingInd)

Ntrain = length(Xtrain);
noblks = Ntrain/tau1;
y1 = reshape(Ytest,tau1,noblks);
yKalman = num2cell(y1,1);

% get model parameters
vary = exp(theta(end));
[Ct,Rt,At,Qt,pIndex] = ...
    getParamsSEHier_single(covfunc,theta(1:end-1),vary,Xtrain,tau1,tau2,tau3,noblks,missingInd);

% initial condition
u1 = splitSpace(Xtrain(1),Xtrain(end),tau2)';
alpha = eval(feval(covfunc{:}));
Kg = feval(covfunc{:},theta(1:alpha),u1) + 1e-7*eye(tau2);
[Xfint,Pfint,Pcovt] = kalmanVarHier(At,Ct,Qt,Rt,pIndex,Kg,yKalman);

fest = zeros(size(Ytest));
vest = zeros(size(Ytest));
u = splitSpace(1,tau1,tau2);
s = 1:tau1;
tin = 1e-6;
hyp.cov = theta(1:end-1);
Kss = feval(covfunc{:},hyp.cov,s') + tin*eye(length(s));
Kuu = feval(covfunc{:},hyp.cov,u') + tin*eye(length(u));
Ksu = feval(covfunc{:},hyp.cov,s',u');
C = Ksu/Kuu;
R = Kss - Ksu/Kuu*Ksu';
for i = 1:noblks
    ind = tau1*(i-1) + (1:tau1);
    fest(ind) = C*Xfint{1}{i};
    vest(ind) = diag(C*Pfint{1}{i}*C' + R);
end
end


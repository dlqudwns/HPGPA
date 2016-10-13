function [fest,vest] = predictSECosMixLocal(theta,K,covfunc,Xtrain,Ytest,tau1,tau2,missingInd)

Ntrain = length(Xtrain);
noblks = Ntrain/tau1;
label = ceil((1:Ntrain)/tau1);
path = 1:noblks;
yKalman = cell(noblks,1);
for i = 1:noblks
    yKalman{i} = Ytest(label==path(i));
end

% get model parameters
covhyp = reshape(theta(1:K*2),2,K)';
mu = exp(theta(K*2+1:end-1));
vary = exp(theta(end));
[Ct,Rt,Qt] = ...
    getParamsSECosMixLocal(covfunc,covhyp,mu,vary,Xtrain,tau1,tau2,noblks,missingInd);

% initial condition
u1 = linspace(Xtrain(1),Xtrain(tau1),tau2)';
x0 = zeros(2*K*tau2,1);
P0 = zeros(2*K*tau2);
for k = 1:K
    Pk = feval(covfunc{:},covhyp(k,:),u1) + 1e-7*eye(tau2);
    P0(2*(k-1)*tau2+(1:tau2),2*(k-1)*tau2+(1:tau2)) = Pk;
    P0((2*k-1)*tau2+(1:tau2),(2*k-1)*tau2+(1:tau2)) = Pk;
end
% keyboard
[Xfint,Pfint] = kalmanVarLocal(Ct,Qt,Rt,x0,P0,yKalman,1,0);

fest = zeros(size(Ytest));
vest = zeros(size(Ytest));

u = linspace(1,tau1,tau2);
s = 1:tau1;
tin = 1e-7;
Ck = cell(K,1);
Rk = cell(K,1);
for k = 1:K
    Kss = feval(covfunc{:},covhyp(k,:),s') + tin*eye(tau1);
    Kuu = feval(covfunc{:},covhyp(k,:),u') + tin*eye(tau2);
    Ksu = feval(covfunc{:},covhyp(k,:),s',u');
    L0 = chol(Kuu);
    Ksu_invL0 = Ksu/L0;
    Ck{k} = Ksu_invL0/L0';
    Rk{k} = Kss - Ksu_invL0*Ksu_invL0';
end

for t = 1:noblks
    ind = tau1*(t-1) + (1:tau1);
    Mt = [];
    Ct = [];
    R = zeros(2*k*tau1);
    for k = 1:K
        Mt1 = diag(cos(2*pi*mu(k)*ind));
        Mt2 = diag(sin(2*pi*mu(k)*ind));
        Mt = [Mt Mt1 Mt2];
        Ct = [Ct Mt1*Ck{k} Mt2*Ck{k}];
        R(2*(k-1)*tau1+(1:tau1),2*(k-1)*tau1+(1:tau1)) = Rk{k};
        R((2*k-1)*tau1+(1:tau1),(2*k-1)*tau1+(1:tau1)) = Rk{k};
    end
    R1 = Mt*R*Mt';
    fest(ind) = Ct*Xfint{t};
    vest(ind) = diag(Ct*Pfint{t}*Ct' + R1);
end
end


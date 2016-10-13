function [f,df] = objFunctionSECosMix(theta,K,covfunc,X,Y,tau1,tau2,missingInd)

% [fest,vest] = predictSECosMix(theta,K,covfunc,X,Y,tau1,tau2,missingInd);
% a = abs(fft(fest));
% figure(10), plot(a(1:length(a)/2))
% xlim([0 4000]);
% drawnow


Ntrain = length(X);
noBlocks = Ntrain/tau1;
label = ceil((1:Ntrain)/tau1);
path = 1:noBlocks;
yKalman = cell(noBlocks,1);
for i = 1:noBlocks
    yKalman{i} = Y(label==path(i));
end




%% get model parameters and the derivatives
noblks = length(yKalman);
covhyp = reshape(theta(1:K*2),2,K)';
mu = exp(theta(K*2+1:end-1));
vary = exp(theta(end));
[Ct,Rt,At,Qt,dCt,dRt,dAt,dQt] = ...
    getParamsSECosMix(covfunc,covhyp,mu,vary,X,tau1,tau2,noblks,missingInd);

% initial condition
u1 = linspace(X(1),X(tau1),tau2)';
x0 = zeros(2*K*tau2,1);
P0 = zeros(2*K*tau2);
for k = 1:K
    Pk = feval(covfunc{:},covhyp(k,:),u1) + 1e-7*eye(tau2);
    P0(2*(k-1)*tau2+(1:tau2),2*(k-1)*tau2+(1:tau2)) = Pk;
    P0((2*k-1)*tau2+(1:tau2),(2*k-1)*tau2+(1:tau2)) = Pk;
end

dP0 = cell(length(theta),1);
for i = 1:length(theta)
    if i>K*2 % if freqs or observation noise
        dP0{i} = zeros(size(P0));
    else
        k = ceil(i/2);
        ind = i - (k-1)*2;
        dP0i = feval(covfunc{:},covhyp(k,:),u1,[],ind);
        dP0k = zeros(size(P0));
        dP0k(2*(k-1)*tau2+(1:tau2),2*(k-1)*tau2+(1:tau2)) = dP0i;
        dP0k((2*k-1)*tau2+(1:tau2),(2*k-1)*tau2+(1:tau2)) = dP0i;
        dP0{i} = dP0k;
    end
end

smoothing = 1;
verbose = 0;
[f,df] = kalmanVar(At,Ct,Qt,Rt,x0,P0,yKalman,verbose,...
    1-smoothing,dAt,dCt,dQt,dRt,dP0);
f = -f/(noblks*tau1);
df = -df/(noblks*tau1);
end


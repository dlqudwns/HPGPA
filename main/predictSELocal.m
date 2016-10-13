function [fest,vest] = predictSELocal(theta,covfunc,Xtrain,Ytest,tau1,tau2,missingInd)

if size(Xtrain, 2) == 1 % 1d
    noblks = size(Xtrain, 1)/tau1;
    y1 = reshape(Ytest,tau1,noblks);
    yKalman = num2cell(y1,1);    
elseif size(Xtrain, 2) == 2 % 2d
    K = (size(Xtrain, 1)/tau1);
    noblks = K^2;
    yKalman = {};
    for j=1:K
        for i=1:K
            yKalman{end+1} = Ytest((i-1)*tau1+1:i*tau1, (j-1)*tau1+1:j*tau1);
            yKalman{end} = yKalman{end}(:);
        end
    end
else
    error('dimension of Xtrain?')
end
% get model parameters
vary = exp(theta(end));
[Ct,Rt,Qt] = ...
    getParamsSELocal(covfunc,theta(1:end-1),vary,Xtrain,tau1,tau2,noblks,missingInd);

% initial condition
if size(Xtrain,2) == 1 %1d
    u1 = linspace(Xtrain(1),Xtrain(tau1),tau2)';
    x0 = zeros(tau2,1);
elseif size(Xtrain,2) == 2 %2d
    u1 = splitSpace([Xtrain(1,1) Xtrain(1,2)], [Xtrain(tau1,1) Xtrain(tau1,2)], tau2)';
    x0 = zeros(tau2*tau2,1);
end

alpha = eval(feval(covfunc{:}));
P0 = feval(covfunc{:},theta(1:alpha),u1);
% keyboard
[Xfint,Pfint] = kalmanVarLocal(Ct,Qt,Rt,x0,P0,yKalman,0,0);

fest = zeros(size(Ytest));
vest = zeros(size(Ytest));

if size(Xtrain, 2) == 1 % 1d
    u = linspace(1,tau1,tau2);
    s = 1:tau1;
    size_s = tau1;
    size_u = tau2;
elseif size(Xtrain, 2) == 2 %2d
    u = splitSpace([Xtrain(1,1) Xtrain(1,2)], [Xtrain(tau1,1) Xtrain(tau1,2)], tau2);
    s = splitSpace([Xtrain(1,1) Xtrain(1,2)], [Xtrain(tau1,1) Xtrain(tau1,2)], tau1);
    size_s = tau1^2;
    size_u = tau2^2;
end

tin = 1e-6;
hyp.cov = theta(1:end-1);
Kss = feval(covfunc{:},hyp.cov,s') + tin*eye(size_s);
Kuu = feval(covfunc{:},hyp.cov,u') + tin*eye(size_u);
Ksu = feval(covfunc{:},hyp.cov,s',u');
C = Ksu/Kuu;
R = Kss - Ksu/Kuu*Ksu';

if size(Xtrain, 2) == 1
    for i = 1:noblks
        ind = tau1*(i-1) + (1:tau1);
        fest(ind) = C*Xfint{i};
        vest(ind) = diag(C*Pfint{i}*C' + R);
    end
elseif size(Xtrain, 2) == 2
    for j=1:sqrt(noblks)
        for i=1:sqrt(noblks)
            ind1 = (i-1)*tau1+1:i*tau1;
            ind2 = (j-1)*tau1+1:j*tau1;
            blk_idx = (j-1)*sqrt(noblks) + i;
            fest(ind1, ind2) = reshape(C*Xfint{blk_idx}, tau1, tau1);
            vest(ind1, ind2) = reshape(diag(C*Pfint{blk_idx}*C' + R), tau1, tau1);
        end
    end
end

end


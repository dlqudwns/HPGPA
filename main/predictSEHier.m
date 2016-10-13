function [fest,vest, Xfint, Pfint] = predictSEHier(theta,covfunc, cross_covfunc,Xtrain,Ytest,tau1,tau2,tau3,missingInd)

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
[Ct,Rt,At,Qt,pIndex] = ...
    getParamsSEHier(covfunc,cross_covfunc,theta(1:end-1),vary,Xtrain,tau1,tau2,tau3,noblks,missingInd);

% initial condition
if size(Xtrain,2) == 1 %1d
    u1 = linspace(Xtrain(1),Xtrain(end),tau2)';
    size_u = tau2;
elseif size(Xtrain,2) == 2 %2d
    u1 = splitSpace([Xtrain(1,1) Xtrain(1,2)], [Xtrain(end,1) Xtrain(end,2)], tau2)';
    size_u = tau2^2;
end
alpha = eval(feval(covfunc{:}));
totthetalen = length(theta);
Kg = feval(covfunc{:},theta(totthetalen-alpha:totthetalen-1),u1) + 1e-7*eye(size_u);
[Xfint,Pfint,Pcovt] = kalmanVarHier(At,Ct,Qt,Rt,pIndex,Kg,yKalman);

fest = zeros(size(Ytest));
vest = zeros(size(Ytest));

if size(Xtrain, 2) == 1 % 1d
    u = linspace(1,tau1,tau2);
    s = 1:tau1;
elseif size(Xtrain, 2) == 2 %2d
    u = splitSpace([Xtrain(1,1) Xtrain(1,2)], [Xtrain(tau1,1) Xtrain(tau1,2)], tau2);
    s = splitSpace([Xtrain(1,1) Xtrain(1,2)], [Xtrain(tau1,1) Xtrain(tau1,2)], tau1);
end
tin = 1e-6;
Kss = feval(covfunc{:},theta(1:2),s')+tin*eye(length(s));
Kuu = feval(covfunc{:},theta(3:4),u')+tin*eye(length(u));
Ksu = feval(cross_covfunc{:},theta(1:2),theta(3:4),s',u');
C = Ksu/Kuu;
R = Kss - Ksu/Kuu*Ksu';

if size(Xtrain, 2) == 1
    for i = 1:noblks
        ind = tau1*(i-1) + (1:tau1);
        fest(ind) = C*Xfint{1}{i};
        vest(ind) = diag(C*Pfint{1}{i}*C' + R);
    end
elseif size(Xtrain, 2) == 2
    for j=1:sqrt(noblks)
        for i=1:sqrt(noblks)
            ind1 = (i-1)*tau1+1:i*tau1;
            ind2 = (j-1)*tau1+1:j*tau1;
            blk_idx = (j-1)*sqrt(noblks) + i;
            fest(ind1, ind2) = reshape(C*Xfint{1}{blk_idx}, tau1, tau1);
            vest(ind1, ind2) = reshape(diag(C*Pfint{1}{blk_idx}*C' + R), tau1, tau1);
        end
    end
end
% 
% for i = 1:noblks
%     ind = tau1*(i-1) + (1:tau1);
%     fest(ind) = C*Xfint{1}{i};
%     vest(ind) = diag(C*Pfint{1}{i}*C' + R);
% end
end


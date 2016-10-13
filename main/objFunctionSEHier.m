function [f,df] = objFunctionSEHier(compressed_theta,covfunc,cross_covfunc,X,Y,tau1,tau2,tau3,missingInd,H)
% expand theta...
if length(compressed_theta) == 2 * H + 1
    theta = compressed_theta;
else
    theta = expand_theta(compressed_theta, H);
end
% theta = compressed_theta;

%% get model parameters and the derivatives
noblks = length(Y);
vary = exp(theta(end));
[Ct,Rt,At,Qt,pIndex,dCt,dRt,dAt,dQt] = ...
    getParamsSEHier(covfunc,cross_covfunc,theta(1:end-1),vary,X,tau1,tau2,tau3,noblks,missingInd);
% initial condition
if size(X, 2) == 1 % 1d
    u1 = splitSpace(X(1),X(end),tau2)';
    size_u = tau2;
elseif size(X, 2) == 2 % 2d
    u1 = splitSpace([X(1) X(1)], [X(end) X(end)], tau2)';
    size_u = tau2^2;
else
    error('dimension error');
end

alpha = eval(feval(covfunc{:}));
totthetalen = length(theta);
Kg = feval(covfunc{:},theta(totthetalen-alpha:totthetalen-1),u1) + 1e-7*eye(size_u);
dKg = cell(totthetalen,1);
for i = 1:alpha
    dP0i = feval(covfunc{:},theta(totthetalen-alpha:totthetalen-1),u1,[],i);
    dKg{i+totthetalen-alpha-1} = dP0i;
end
for i=1:totthetalen
    if isempty(dKg{i})
        dKg{i} = zeros(size(Kg));
    end
end

[f,df] = kalmanVarHier(At,Ct,Qt,Rt,pIndex,Kg,Y,...
    dAt,dCt,dQt,dRt,dKg);
f = -f/(noblks*tau1);
df = -df/(noblks*tau1);

if length(compressed_theta) < 2 * H + 1
    % for compressed theta
    df = [sum(df(1:2:end-1)) ; df(2) ; df(end)];
end


fprintf('f: %f\n', f);
for i=1:length(theta)
    fprintf('%6.3f ', theta(i));
end
fprintf('\n');
for i=1:length(df)
    fprintf('%6.3f ', df(i));
end
fprintf('\n');
end


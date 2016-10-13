function [f,df] = objFunctionSE(theta,covfunc,X,Y,tau1,tau2,missingInd,adj_matrix)


%% get model parameters and the derivatives
noblks = length(Y);
vary = exp(theta(end));


if size(X,2) == 1 %1d
    [Ct,Rt,At,Qt,dCt,dRt,dAt,dQt] = ...
    getParamsSE(covfunc,theta(1:end-1),vary,X,tau1,tau2,noblks,missingInd);
    u1 = linspace(X(1),X(tau1),tau2)';
    x0 = zeros(tau2,1);
elseif size(X,2) == 2 %2d
    [Ct,Rt,At,Qt,indexer,dCt,dRt,dAt,dQt] = ...
    getParamsSETree(covfunc,theta(1:end-1),vary,X,tau1,tau2,noblks,missingInd,adj_matrix);
    u1 = splitSpace([X(1,1) X(1,2)], [X(tau1,1) X(tau1,2)], tau2)';
    x0 = zeros(tau2*tau2,1);
end


% initial condition
alpha = eval(feval(covfunc{:}));
P0 = feval(covfunc{:},theta(1:alpha),u1);
dP0 = cell(alpha+1,1);
for i = 1:alpha
    dP0i = feval(covfunc{:},theta(1:alpha),u1,[],i);
    dP0{i} = dP0i;
end
dP0{end} = zeros(size(P0));
smoothing = 1;
verbose = 0;
if size(X,2) == 1 %1d
    [f,df] = kalmanVar(At,Ct,Qt,Rt,x0,P0,Y,verbose,...
        1-smoothing,dAt,dCt,dQt,dRt,dP0);
else
    [f,df] = kalmanVarTree(At,Ct,Qt,Rt,indexer,P0,Y,...
        dAt,dCt,dQt,dRt,dP0);
end
f = -f/(noblks*tau1);
df = -df/(noblks*tau1);

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


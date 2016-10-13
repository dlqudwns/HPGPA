function [Ct,Rt,Qt,dCt,dRt,dQt] = getParamsSELocal(covfunc,hyp,vary,X,tau1,tau2,noBlks,missingInd)

tiny = 1e-8;
sigmaInf = 100;
if size(X, 2) == 1 %1d
    s = (X(1:tau1));
    u = linspace(X(1),X(tau1),tau2)';
    size_s = tau1;
    size_u = tau2;
elseif size(X, 2) == 2 % 2d
    s = splitSpace([1 1], [tau1 tau1], tau1)';
    u = splitSpace([X(1,1) X(1,2)], [X(tau1,1) X(tau1,2)], tau2)';
    size_s = tau1 * tau1;
    size_u = tau2 * tau2;
else
    error('dimension error');
end

Kss = feval(covfunc{:},hyp,s)+tiny*eye(size_s);
Kuu = feval(covfunc{:},hyp,u)+tiny*eye(size_u);
Ksu = feval(covfunc{:},hyp,s,u);

L0 = chol(Kuu);
Ksu_invL0 = Ksu/L0;
C = Ksu_invL0/L0';
R = Kss - Ksu_invL0*Ksu_invL0' + tiny*eye(size_s);
Q = Kuu + tiny*eye(size_u);
T = noBlks;
Ct = repmat({C},T,1);
Rt = repmat({R},T,1);
Qt = repmat({Q},T,1);

if size(X, 2) == 1 %1d
    for t = 1:T
        mInd = missingInd(t,:);
        sigma2 = vary*ones(size_s,1);
        sigma2(mInd) = sigma2(mInd)+sigmaInf; % set large variance for missing indices
        Rt{t} = R + diag(sigma2);
    end
elseif size(X, 2) == 2 %2d
    for t = 1:T
        mInd = missingInd{t};
        sigma2 = vary*ones(size_s,1);
        sigma2(mInd(:)) = sigma2(mInd(:)) + sigmaInf; % set large variance for missing indices
        Rt{t} = R + diag(sigma2);
    end
end

if nargout>3
    noVar = length(hyp)+1; 
    dC = cell(noVar,1);
    dR = cell(noVar,1);
    dQ = cell(noVar,1);
    for j = 1:noVar
        if j == noVar
            dCj = zeros(size(C));
            dRj = vary*eye(size(R));
            dQj = zeros(length(Q));
        else
            Kssdi = feval(covfunc{:},hyp,s,[],j);
            Kuudi = feval(covfunc{:},hyp,u,[],j);
            Ksudi = feval(covfunc{:},hyp,s,u,j);
            Ka = Kuudi/Kuu;
            Kb = Ksudi/Kuu;

            dCj = Kb - C*Ka;
            dRj = Kssdi - Kb*Ksu' + C*Ka*Ksu' - C*Ksudi';

            dQj = Kuudi;
        end
        dC{j} = dCj;
        dR{j} = dRj;
        dQ{j} = dQj;
    end
    
    dCt = cell(T,noVar);
    dRt = cell(T,noVar);
    dQt = cell(T,noVar);
    for j = 1:noVar
        [dQt{1:T,j}] = deal(dQ{j});
        [dCt{1:T,j}] = deal(dC{j});
        [dRt{1:T,j}] = deal(dR{j});
    end

end


end
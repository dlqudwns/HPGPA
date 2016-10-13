function [Ct,Rt,At,Qt,dCt,dRt,dAt,dQt] = getParamsSE(covfunc,hyp,vary,X,tau1,tau2,noBlks,missingInd)
   

sigmaInf = 100;
s = (X(1:tau1));
u = linspace(X(1),X(tau1),tau2)';
tiny = 1e-8;
Kss = feval(covfunc{:},hyp,s)+tiny*eye(tau1);
Kuu = feval(covfunc{:},hyp,u)+tiny*eye(tau2);
Ksu = feval(covfunc{:},hyp,s,u);

L0 = chol(Kuu);
Ksu_invL0 = Ksu/L0;
C = Ksu_invL0/L0';
R = Kss - Ksu_invL0*Ksu_invL0' + tiny*eye(tau1);
uprev = u-X(tau1);
Kupup = feval(covfunc{:}, hyp, uprev)+tiny*eye(tau2);
Kuup  = feval(covfunc{:}, hyp, u, uprev);

L1 = chol(Kupup);
Kuup_invL1 = Kuup/L1;
A = Kuup_invL1/L1';
Q = Kuu - Kuup_invL1*Kuup_invL1'+tiny*eye(tau2);
T = noBlks;
Ct = repmat({C},T,1);
Rt = repmat({R},T,1);
At = repmat({A},T,1);
Qt = repmat({Q},T,1);
for t = 1:T
    %ind = tau1*(t-1) + (1:tau1);
    mInd = [];
    if ~isempty(missingInd)
        mInd = missingInd(t,:);
    end
    sigma2 = vary*ones(tau1,1);
    sigma2(mInd) = sigma2(mInd)+sigmaInf; % set large variance for missing indices
    Rt{t} = R + diag(sigma2);
end

if nargout>4
    noVar = length(hyp)+1; 
    dC = cell(noVar,1);
    dR = cell(noVar,1);
    dA = cell(noVar,1);
    dQ = cell(noVar,1);
    for j = 1:noVar
        if j == noVar
            dCj = zeros(size(C));
            dRj = vary*eye(size(R));
           
            dAj = zeros(length(A));
            dQj = zeros(length(Q));
        else
            Kssdi = feval(covfunc{:},hyp,s,[],j);
            Kuudi = feval(covfunc{:},hyp,u,[],j);
            Ksudi = feval(covfunc{:},hyp,s,u,j);
            Ka = Kuudi/Kuu;
            Kb = Ksudi/Kuu;

            dCj = Kb - C*Ka;
            dRj = Kssdi - Kb*Ksu' + C*Ka*Ksu' - C*Ksudi';
            
            Kupupdi = feval(covfunc{:},hyp,uprev,[],j);
            Kuupdi = feval(covfunc{:},hyp,u,uprev,j);
            Ka = Kupupdi/Kupup;
            Kb = Kuupdi/Kupup;
                    
            dAj = Kb - A*Ka;
            dQj = Kuudi - Kb*Kuup' + A*Ka*Kuup' - A*Kuupdi';
        end
        dC{j} = dCj;
        dR{j} = dRj;
        dA{j} = dAj;
        dQ{j} = dQj;
    end
    
    dCt = cell(T,noVar);
    dRt = cell(T,noVar);
    dAt = cell(T,noVar);
    dQt = cell(T,noVar);
    for j = 1:noVar
        [dAt{1:T,j}] = deal(dA{j});
        [dQt{1:T,j}] = deal(dQ{j});
        [dCt{1:T,j}] = deal(dC{j});
        [dRt{1:T,j}] = deal(dR{j});
    end

end

end
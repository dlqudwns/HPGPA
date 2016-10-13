function [Ct,Rt,At,Qt,dCt,dRt,dAt,dQt] = getParamsSECosMix(covfunc,hyp,mu,vary,X,tau1,tau2,noBlks,missingInd)
% extract dynamical system weights for a uniformly sampled dataset

K = size(hyp,1); % number of components

sigmaInf = 100;
s = (X(1:tau1));
u = linspace(X(1),X(tau1),tau2)';
uprev = u-X(tau1);
tiny = 1e-7;
Ck = cell(K,1);
Rk = cell(K,1);
Ak = cell(K,1);
Qk = cell(K,1);

% find the transition and emission dynamics for each component
Kuuk = cell(K,1);
Ksuk = cell(K,1);
Kupupk = cell(K,1);
Kuupk = cell(K,1);
for k = 1:K
    hypk = hyp(k,:);
    Kss = feval(covfunc{:},hypk,s)+tiny*eye(tau1);
    Kuu = feval(covfunc{:},hypk,u)+tiny*eye(tau2);
    Ksu = feval(covfunc{:},hypk,s,u);
    Kuuk{k} = Kuu;
    Ksuk{k} = Ksu;
    
    L0 = chol(Kuu);
    Ksu_invL0 = Ksu/L0;
    Ck{k} = Ksu_invL0/L0';
    %Ck{k} = ones(size(Ck{k}));
    Rk{k} = Kss - Ksu_invL0*Ksu_invL0';
    
    Kupup = feval(covfunc{:}, hypk, uprev)+tiny*eye(tau2);
    Kuup  = feval(covfunc{:}, hypk, u, uprev);
    Kupupk{k} = Kupup;
    Kuupk{k} = Kuup;
    L1 = chol(Kupup);
    Kuup_invL1 = Kuup/L1;
    Ak{k} = Kuup_invL1/L1';
    Qk{k} = Kuu - Kuup_invL1*Kuup_invL1' + tiny*eye(tau2);
end

T = noBlks;
A1 = zeros(2*K*tau2);
Q1 = zeros(2*K*tau2);
for k = 1:K
    A1(2*(k-1)*tau2+(1:tau2),2*(k-1)*tau2+(1:tau2)) = Ak{k};
    A1((2*k-1)*tau2+(1:tau2),(2*k-1)*tau2+(1:tau2)) = Ak{k};
    Q1(2*(k-1)*tau2+(1:tau2),2*(k-1)*tau2+(1:tau2)) = Qk{k};
    Q1((2*k-1)*tau2+(1:tau2),(2*k-1)*tau2+(1:tau2)) = Qk{k};
end
At = repmat({A1},T,1);
Qt = repmat({Q1},T,1);


Ct = cell(T,1);
Rt = cell(T,1);
Pt = cell(T,1);
Mt1 = cell(T,K);
Mt2 = cell(T,K);
Mt = cell(T,1);

for t = 1:T
    ind = tau1*(t-1) + (1:tau1);
    Mt{t} = [];
    Ct{t} = [];
    R = zeros(2*k*tau1);
    for k = 1:K
        Mt1{t,k} = diag(cos(2*pi*mu(k)*ind));
        Mt2{t,k} = diag(sin(2*pi*mu(k)*ind));
        
%         Mt1{t,k} = diag(cos(2*pi*5*ind));
%         Mt2{t,k} = diag(sin(2*pi*5*ind));
        
        Mt{t} = [Mt{t} Mt1{t,k} Mt2{t,k}];
        Ct{t} = [Ct{t} Mt1{t,k}*Ck{k} Mt2{t,k}*Ck{k}];
        R(2*(k-1)*tau1+(1:tau1),2*(k-1)*tau1+(1:tau1)) = Rk{k};
        R((2*k-1)*tau1+(1:tau1),(2*k-1)*tau1+(1:tau1)) = Rk{k};
    end
    Pt{t} = R;
    R1 = Mt{t}*R*Mt{t}';
    mInd = missingInd(t,:);
    sigma2 = vary*ones(tau1,1);
    sigma2(mInd) = sigma2(mInd)+sigmaInf; % set large variance for missing indices
    Rt{t} = R1 + diag(sigma2);
end

if nargout>4
    % numel(hyp) should be k*2
    D = numel(hyp);
    noVar = D+length(mu)+1; % SE hypers, freqs and obs noise
    dC = cell(noVar,1);
    dR = cell(noVar,1);
    dA = cell(noVar,1);
    dQ = cell(noVar,1);
    for j = 1:noVar
        if j > D
            dCj = zeros(tau1,tau2);
            dRj = zeros(2*K*tau1);
            dAj = zeros(2*K*tau2);
            dQj = zeros(2*K*tau2);
        else
            c = ceil(j/2); % spectral component c
            paramInd = j - (c-1)*2; % param index
            hypc = hyp(c,:);
            Kssdi = feval(covfunc{:},hypc,s,[],paramInd);
            Kuudi = feval(covfunc{:},hypc,u,[],paramInd);
            Ksudi = feval(covfunc{:},hypc,s,u,paramInd);
            Ka = Kuudi/Kuuk{c};
            Kb = Ksudi/Kuuk{c};
            dCj = Kb - Ck{c}*Ka;
            %dCj = zeros(size(dCj));
            
            dR1 = Kssdi - Kb*Ksuk{c}' + Ck{c}*Ka*Ksuk{c}' - Ck{c}*Ksudi';
            dRj = zeros(2*K*tau1);
            dRj(2*(c-1)*tau1+(1:tau1),2*(c-1)*tau1+(1:tau1)) = dR1;
            dRj((2*c-1)*tau1+(1:tau1),(2*c-1)*tau1+(1:tau1)) = dR1;
            Kupupdi = feval(covfunc{:},hypc,uprev,[],paramInd);
            Kuupdi = feval(covfunc{:},hypc,u,uprev,paramInd);
            Ka = Kupupdi/Kupupk{c};
            Kb = Kuupdi/Kupupk{c};
            dA1 = Kb - Ak{c}*Ka;
            dQ1 = Kuudi - Kb*Kuupk{c}' + Ak{c}*Ka*Kuupk{c}' - Ak{c}*Kuupdi';
            
            dAj = zeros(2*K*tau2);
            dQj = zeros(2*K*tau2);
            dAj(2*(c-1)*tau2+(1:tau2),2*(c-1)*tau2+(1:tau2)) = dA1;
            dAj((2*c-1)*tau2+(1:tau2),(2*c-1)*tau2+(1:tau2)) = dA1;
            dQj(2*(c-1)*tau2+(1:tau2),2*(c-1)*tau2+(1:tau2)) = dQ1;
            dQj((2*c-1)*tau2+(1:tau2),(2*c-1)*tau2+(1:tau2)) = dQ1;
        end
        dC{j} = dCj; % tau1 x tau2
        dR{j} = dRj; % 2*K*tau1 x 2*K*tau1
        dA{j} = dAj; % 2*K*tau2 x 2*K*tau2
        dQ{j} = dQj; % 2*K*tau2 x 2*K*tau2
    end
    
    dCt = cell(T,noVar);
    dRt = cell(T,noVar);
    dAt = cell(T,noVar);
    dQt = cell(T,noVar);
    for j = 1:noVar
        [dAt{1:T,j}] = deal(dA{j});
        [dQt{1:T,j}] = deal(dQ{j});
    end
    
    for t = 1:T
        for j = 1:noVar
            if j == noVar % observation noise
                dCt{t,j} = zeros(tau1,2*K*tau2);
                sigma2 = vary*ones(tau1,1);
                dRt{t,j} = diag(sigma2);
            elseif j>D && j<=noVar-1 % frequencies
                c = j-D; % spectral component c
                ind = tau1*(t-1) + (1:tau1);
                pi2mut = 2*pi*mu(c)*ind;
                
                dMt1 = diag(-pi2mut.*sin(pi2mut));
                dMt2 = diag(pi2mut.*cos(pi2mut));
                
                dCtj = zeros(tau1,2*K*tau2);
                dCtj(:,2*(c-1)*tau2+(1:2*tau2)) = [dMt1*Ck{c} dMt2*Ck{c}];
                dCt{t,j} = dCtj;
                dMt = zeros(tau1,2*K*tau1);
                dMt(:,2*(c-1)*tau1+(1:2*tau1)) = [dMt1 dMt2];
                dRt{t,j} = dMt*Pt{t}*Mt{t}' + Mt{t}*Pt{t}*dMt';
            else % other params
                c = ceil(j/2); % spectral component c
                dCtj = zeros(tau1,2*K*tau2);
                dCtj(:,2*(c-1)*tau2+(1:2*tau2)) = [Mt1{t,c}*dC{j} Mt2{t,c}*dC{j}];
                dCt{t,j} = dCtj;
                dRt{t,j} = Mt{t}*dR{j}*Mt{t}';
            end
        end
        
    end
    
end
end
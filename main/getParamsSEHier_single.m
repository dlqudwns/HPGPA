function [Ct,Rt,At,Qt,pIndex,dCt,dRt,dAt,dQt] = getParamsSEHier_single(covfunc,hyp,vary,X,tau1,tau2,tau3,noBlks,missingInd)
   


sigmaInf = 100;
s = (X(1:tau1));
u = splitSpace(X(1),X(tau1),tau2)';
tiny = 1e-8;
Kss = feval(covfunc{:},hyp,s)+tiny*eye(tau1);
Kuu = feval(covfunc{:},hyp,u)+tiny*eye(tau2);
Ksu = feval(covfunc{:},hyp,s,u);

L0 = chol(Kuu);
Ksu_invL0 = Ksu/L0;
C = Ksu_invL0/L0';
R = Kss - Ksu_invL0*Ksu_invL0' + tiny*eye(tau1);

T = noBlks;
H = floor(log(noBlks)/log(tau3))+1;
Ct = cell(H,1);
Rt = cell(H,1);
Ct{1} = repmat({C},T,1);
Rt{1} = repmat({R},T,1);

At = cell(H,1);
Qt = cell(H,1);
cur_tau1 = tau1;
norm_blks = noBlks;
pIndex = cell(H,1);
pIndex2 = cell(H,1);
pIndex{1} = 1:T;
for h = 1:H-1
    uparent = splitSpace(X(1),X(cur_tau1 * tau3),tau2)';
    
    Atau = cell(tau3,1);
    Qtau = cell(tau3,1);
    for tau3_index = 1:tau3
        cur_u = splitSpace(X(1 + cur_tau1 * (tau3_index - 1)),X(cur_tau1 * tau3_index),tau2)';
        Kupup = feval(covfunc{:}, hyp, uparent)+tiny*eye(tau2);
        Kuup  = feval(covfunc{:}, hyp, cur_u, uparent);
        Kuu  = feval(covfunc{:}, hyp, cur_u)+tiny*eye(tau2);
        L1 = chol(Kupup);
        Kuup_invL1 = Kuup/L1;
        Atau{tau3_index} = Kuup_invL1/L1';
        Qtau{tau3_index} = Kuu - Kuup_invL1*Kuup_invL1'+tiny*eye(tau2);
    end
    last_blocks = tau3 + mod(norm_blks,tau3);
    Atau_last = cell(last_blocks,1);
    Qtau_last = cell(last_blocks,1);
    uparent = splitSpace(X(1),X(cur_tau1 * last_blocks),tau2)';
    for tau3_index = 1:last_blocks
        cur_u = splitSpace(X(1 + cur_tau1 * (tau3_index - 1)),X(cur_tau1 * tau3_index),tau2)';
        Kupup = feval(covfunc{:}, hyp, uparent)+tiny*eye(tau2);
        Kuup  = feval(covfunc{:}, hyp, cur_u, uparent);
        Kuu  = feval(covfunc{:}, hyp, cur_u)+tiny*eye(tau2);
        L1 = chol(Kupup);
        Kuup_invL1 = Kuup/L1;
        Atau_last{tau3_index} = Kuup_invL1/L1';
        Qtau_last{tau3_index} = Kuu - Kuup_invL1*Kuup_invL1'+tiny*eye(tau2);
    end
    pIndex2{h} = ceil((1:norm_blks)/tau3);
    if sum(pIndex2{h} == pIndex2{h}(end)) < tau3
        pIndex2{h}(pIndex2{h} == pIndex2{h}(end)) = pIndex2{h}(end) - 1;
    end
    
    norm_blks = floor(norm_blks / tau3);
    At{h} = [repmat(Atau,norm_blks-1,1);Atau_last];
    Qt{h} = [repmat(Qtau,norm_blks-1,1);Qtau_last];
    cur_tau1 = cur_tau1 * tau3;
    pIndex{h+1} = ceil(pIndex{h}/tau3);
    if sum(pIndex{h+1} == pIndex{h+1}(end)) < tau3^h
        pIndex{h+1}(pIndex{h+1} == pIndex{h+1}(end)) = pIndex{h+1}(end) - 1;
    end
end
pIndex = {pIndex;pIndex2};
for t = 1:T
    %ind = tau1*(t-1) + (1:tau1);
    mInd = [];
    if ~isempty(missingInd)
        mInd = missingInd(t,:);
    end
    sigma2 = vary*ones(tau1,1);
    sigma2(mInd) = sigma2(mInd)+sigmaInf; % set large variance for missing indices
    Rt{1}{t} = R + diag(sigma2);
end

if nargout>5
    noVar = length(hyp)+1; 
    dC = cell(noVar,1);
    dR = cell(noVar,1);
    for j = 1:noVar
        if j == noVar
            dCj = zeros(size(C));
            dRj = vary*eye(size(R));
        else
            Kuudi = feval(covfunc{:},hyp,u,[],j);
            Kuu = feval(covfunc{:},hyp,u)+tiny*eye(tau2);
            Ksudi = feval(covfunc{:},hyp,s,u,j);
            Ksu = feval(covfunc{:},hyp,s,u);
            Kssdi = feval(covfunc{:},hyp,s,[],j);
            Ka = Kuudi/Kuu;
            Kb = Ksudi/Kuu;
            dCj = Kb -  C*Ka;
            dRj = Kssdi -Kb*Ksu' + C*Ka*Ksu' - C*Ksudi';
        end
        dC{j} = dCj;
        dR{j} = dRj;
    end
    dCt = cell(T,noVar);
    dRt = cell(T,noVar);
    for j = 1:noVar
        [dCt{1:T,j}] = deal(dC{j});
        [dRt{1:T,j}] = deal(dR{j});
    end
    
    dAt = cell(H,noVar);
    dQt = cell(H,noVar);
    cur_tau1 = tau1;
    norm_blks = noBlks;
    for h = 1:H-1

        blk_group_no = floor(norm_blks / tau3) - 1;
        uparent = splitSpace(X(1),X(cur_tau1 * tau3),tau2)';
        uparent_l = splitSpace(X(1),X(cur_tau1 * last_blocks),tau2)';
        for j = 1:noVar
            if j == noVar
                dAt{h,j} = repmat({zeros(size(At{h}{1}))},length(At{h}),1);
                dQt{h,j} = repmat({zeros(size(At{h}{1}))},length(At{h}),1);
            else
                Atau = cell(tau3,1);
                Qtau = cell(tau3,1);
                for tau3_index = 1:tau3
                    ori_At = At{h}{tau3_index};
                    cur_u = splitSpace(X(1 + cur_tau1 * (tau3_index - 1)),X(cur_tau1 * tau3_index),tau2)';
                    Kupupdi = feval(covfunc{:}, hyp, uparent,[],j);
                    Kuupdi  = feval(covfunc{:}, hyp, cur_u, uparent,j);
                    Kuup  = feval(covfunc{:}, hyp, cur_u, uparent);
                    Kupup = feval(covfunc{:}, hyp, uparent)+tiny*eye(tau2);
                    Kuudi = feval(covfunc{:}, hyp,cur_u,[],j);
                    Ka = Kupupdi/Kupup;
                    Kb = Kuupdi/Kupup;
                    Atau{tau3_index} = Kb - ori_At*Ka;
                    Qtau{tau3_index} = Kuudi - Kb*Kuup' + ori_At*Ka*Kuup' - ori_At*Kuupdi';
                end
                last_blocks = tau3 + mod(norm_blks,tau3);
                Atau_last = cell(last_blocks,1);
                Qtau_last = cell(last_blocks,1);
                for tau3_index = 1:last_blocks
                    ori_At = At{h}{end-last_blocks+tau3_index};
                    cur_u = splitSpace(X(1 + cur_tau1 * (tau3_index - 1)),X(cur_tau1 * tau3_index),tau2)';
                    Kupupdi = feval(covfunc{:}, hyp, uparent_l,[],j);
                    Kuupdi  = feval(covfunc{:}, hyp, cur_u, uparent_l,j);
                    Kuup  = feval(covfunc{:}, hyp, cur_u, uparent_l);
                    Kupup = feval(covfunc{:}, hyp, uparent_l)+tiny*eye(tau2);
                    Kuudi = feval(covfunc{:}, hyp,cur_u,[],j);
                    Ka = Kupupdi/Kupup;
                    Kb = Kuupdi/Kupup;
                    Atau_last{tau3_index} = Kb - ori_At*Ka;
                    Qtau_last{tau3_index} = Kuudi - Kb*Kuup' + ori_At*Ka*Kuup' - ori_At*Kuupdi';
                end
                dAt{h,j} = [repmat(Atau,blk_group_no,1);Atau_last];
                dQt{h,j} = [repmat(Qtau,blk_group_no,1);Qtau_last];
            end
        end

        norm_blks = floor(norm_blks / tau3);
        cur_tau1 = cur_tau1 * tau3;
    end


end

end
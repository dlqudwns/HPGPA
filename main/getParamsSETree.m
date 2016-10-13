function [Ct,Rt,At,Qt,indexer,dCt,dRt,dAt,dQt] = getParamsSETree(covfunc,hyp,vary,X,tau1,tau2,noBlks,missingInd,adj_matrix)
   
K = length(X)/tau1;
root = floor(noBlks/2);

sigmaInf = 100;
tiny = 1e-8;

s = splitSpace([1 1], [tau1 tau1], tau1)';
u = splitSpace([X(1,1) X(1,2)], [X(tau1,1) X(tau1,2)], tau2)';
size_s = tau1 * tau1;
size_u = tau2 * tau2;

Kss = feval(covfunc{:},hyp,s)+tiny*eye(size_s);
Kuu = feval(covfunc{:},hyp,u)+tiny*eye(size_u);
Ksu = feval(covfunc{:},hyp,s,u);

L0 = chol(Kuu);
Ksu_invL0 = Ksu/L0;
C = Ksu_invL0/L0';
R = Kss - Ksu_invL0*Ksu_invL0' + tiny*eye(size_s);

blocks_processed = 0;
cur_blocks = (root);
par_blocks = [];
pid_blocks = [];
adj_matrix(:,root) = 0;

Ct = {};Rt = {};At = {};Qt = {};
if nargout>5
    dCt = {};dRt = {};dAt = {};dQt = {};
end
tree = {};
invtree = zeros(noBlks,2);
pIndex = {};
height = 1;

noVar = length(hyp)+1; 
while blocks_processed < noBlks
    T = length(cur_blocks);
    Ct{end+1} = repmat({C},T,1); Rt{end+1} = repmat({R},T,1);
    At{end+1} = cell(T,1); Qt{end+1} = cell(T,1);
    if nargout>5
        dCt{end+1} = cell(T,noVar);dRt{end+1} = cell(T,noVar);
        dAt{end+1} = cell(T,noVar);dQt{end+1} = cell(T,noVar);
        dC = cell(noVar,1); dR = cell(noVar,1);
        for j = 1:noVar
            if j == noVar
                dCj = zeros(size(C));
                dRj = vary*eye(size(R));
                if isempty(par_blocks)
                    dQt{end}{1,j} = zeros(size_u);
                end
            else
                Kuudi = feval(covfunc{:},hyp,u,[],j);
                Kuu = feval(covfunc{:},hyp,u)+tiny*eye(size_u);
                Ksudi = feval(covfunc{:},hyp,s,u,j);
                Ksu = feval(covfunc{:},hyp,s,u);
                Kssdi = feval(covfunc{:},hyp,s,[],j);
                Ka = Kuudi/Kuu;
                Kb = Ksudi/Kuu;
                dCj = Kb -  C*Ka;
                dRj = Kssdi -Kb*Ksu' + C*Ka*Ksu' - C*Ksudi';
                if isempty(par_blocks)
                    dQt{end}{1,j} = Kuudi + tiny*eye(size_u);
                end
            end
            dC{j} = dCj; dR{j} = dRj;
        end
        for j = 1:noVar
            [dCt{end}{1:T,j}] = deal(dC{j});[dRt{end}{1:T,j}] = deal(dR{j});
        end
    end
    pIndex{end+1} = pid_blocks; tree{end+1} = cur_blocks;
    for i = 1:T
        b = cur_blocks(i);
        invtree(b,:) = [height,i];
        if isempty(par_blocks)
            Qt{end}{1} = Kuu + tiny*eye(size_u);
        else
            p = par_blocks(i);
            bi = mod(b,K);
            bj = floor(b/K);
            startInd = [(bi-1)*tau1+1, (bj-1)*tau1+1];
            endInd = [tau1*bi, tau1*bj];
            
            pi = mod(p,K);
            pj = floor(p/K);
            pstartInd = [(pi-1)*tau1+1, (pj-1)*tau1+1];
            pendInd = [tau1*pi, tau1*pj];
            
            uu = splitSpace(startInd, endInd, tau2)';
            upar = splitSpace(pstartInd, pendInd, tau2)';
            
            Kupup = feval(covfunc{:}, hyp, upar)+tiny*eye(size_u);
            Kuup  = feval(covfunc{:}, hyp, uu, upar);
            L1 = chol(Kupup);
            Kuup_invL1 = Kuup/L1;
            At{end}{i} = Kuup_invL1/L1';
            Qt{end}{i} = Kuu - Kuup_invL1*Kuup_invL1'+tiny*eye(size_u);
            
            if nargout>5
                for j = 1:noVar
                    if j == noVar
                        dAt{end}{i,j} = zeros(size_u);
                        dQt{end}{i,j} = zeros(size_u);
                    else
                        ori_At = At{end}{i};
                        Kupupdi = feval(covfunc{:}, hyp, upar,[],j);
                        Kuupdi  = feval(covfunc{:}, hyp, uu, upar,j);
                        Kuudi = feval(covfunc{:}, hyp, uu,[],j);

                        Ka = Kupupdi/Kupup;
                        Kb = Kuupdi/Kupup;
                        dAt{end}{i,j} = Kb - ori_At*Ka;
                        dQt{end}{i,j} = Kuudi - Kb*Kuup' + ori_At*Ka*Kuup' - ori_At*Kuupdi';
                    end
                end
            end
            
        end
    end
        
    blocks_processed = blocks_processed + T;
    %% spanning tree child indexing
    childs = [];
    par_blocks = [];
    pid_blocks = [];
    for i = 1:T
        number_index = 1:noBlks;
        new_childs = number_index(adj_matrix(cur_blocks(i),:)==1);
        childs = [childs new_childs];
        parents = ones(size(new_childs)) * cur_blocks(i);
        par_blocks = [par_blocks parents];
        parent_index = ones(size(new_childs)) * i;
        pid_blocks = [pid_blocks parent_index];
        adj_matrix(:,new_childs) = 0;
    end
    cur_blocks = childs;
    height = height + 1;
end

for t = 1:noBlks
    mInd = missingInd{t};
    sigma2 = vary*ones(size_s,1);
    sigma2(mInd(:)) = sigma2(mInd(:)) + sigmaInf; % set large variance for missing indices
    Rt{invtree(t,1)}{invtree(t,2)} = Rt{invtree(t,1)}{invtree(t,2)} + diag(sigma2);
end
indexer.pindex = pIndex;
indexer.tree = tree;
indexer.invtree = invtree;
end


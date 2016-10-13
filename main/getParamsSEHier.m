function [Ct,Rt,At,Qt,pIndex,dCt,dRt,dAt,dQt] = getParamsSEHier(covfunc,cross_covfunc,hyp,vary,X,tau1,tau2,tau3,noBlks,missingInd)

alpha = eval(feval(covfunc{:}));
noHypSet = length(hyp)/alpha;
hyp = reshape(hyp,alpha,noHypSet)';

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

Kss = feval(covfunc{:},hyp(1,:),s)+tiny*eye(size_s);
Kuu = feval(covfunc{:},hyp(2,:),u)+tiny*eye(size_u);
Ksu = feval(cross_covfunc{:},hyp(1,:),hyp(2,:),s,u);

L0 = chol(Kuu);
Ksu_invL0 = Ksu/L0;
C = Ksu_invL0/L0';
R = Kss - Ksu_invL0*Ksu_invL0' + tiny*eye(size_s);

T = noBlks;
if size(X, 2) == 1 %1d
    H = computeH(noBlks, tau3);
    group_factor = tau3;
elseif size(X, 2) == 2 %2d
    H = computeH(noBlks, tau3^2);
    group_factor = tau3^2;
else
    error('dimension error');
end
Ct = cell(H,1);
Rt = cell(H,1);
Ct{1} = repmat({C},T,1);
Rt{1} = repmat({R},T,1);

At = cell(H,1);
Qt = cell(H,1);
cur_tau1 = tau1;
last_block_tau1 = cur_tau1;
norm_blks = noBlks;
pIndex2 = cell(H,1);
for h = 1:H-1
    if size(X, 2) == 1 % 1d
        toIdx = min(cur_tau1 * tau3, X(end));
        uparent = splitSpace(X(1), X(toIdx), tau2)';
    elseif size(X, 2) == 2 % 2d
        toIdx = min(cur_tau1 * tau3, X(end, 1));
        uparent = splitSpace([X(1,1) X(1,2)], [X(toIdx,1) X(toIdx,2)], tau2)';
    end
    
    Atau = cell(group_factor,1);
    Qtau = cell(group_factor,1);
    for tau3_index = 1:min(group_factor, floor(X(end) / cur_tau1)) % loop for subblock
        if size(X, 2) == 1 % 1d
            fromIdx = 1 + cur_tau1 * (tau3_index - 1);
            toIdx = cur_tau1 * tau3_index;
            cur_u = splitSpace(X(fromIdx), X(toIdx), tau2)';
        elseif size(X, 2) == 2 % 2d
            % if uparent =
            % [ a b ;
            %   c d ]
            % , then cur_u = a, b, c, or d.
            row = mod(tau3_index-1, tau3) + 1;
            col = floor((tau3_index-1) / tau3) + 1;
            fromIdx = 1 + [cur_tau1 * (row-1) cur_tau1 * (col-1)];
            toIdx = [cur_tau1 * row cur_tau1 * col];
            cur_u = splitSpace([X(fromIdx, 1) X(fromIdx, 2)], [X(toIdx, 1) X(toIdx, 2)], tau2)';
        end

        Kupup = feval(covfunc{:}, hyp(h+2,:), uparent) + tiny*eye(size_u);
        Kuup  = feval(cross_covfunc{:}, hyp(h+1,:), hyp(h+2,:), cur_u, uparent);
        Kuu  = feval(covfunc{:}, hyp(h+1,:), cur_u) + tiny*eye(size_u);
        L1 = chol(Kupup);
        Kuup_invL1 = Kuup/L1;
        Atau{tau3_index} = Kuup_invL1/L1';
        Qtau{tau3_index} = Kuu - Kuup_invL1*Kuup_invL1' + tiny*eye(size_u);
    end
    if norm_blks > group_factor
        last_blocks = group_factor + mod(norm_blks, group_factor);
    else
        last_blocks = norm_blks;
    end
    Atau_last = cell(last_blocks,1);
    Qtau_last = cell(last_blocks,1);
    
    if size(X, 2) == 1 % 1d
        toIdx = cur_tau1 * (last_blocks-1) + last_block_tau1;
        uparent = splitSpace(X(1),X(toIdx),tau2)';
        last_block_tau1 = cur_tau1 * (last_blocks-1) + last_block_tau1;
    elseif size(X, 2) == 2 % 2d
        toIdx = cur_tau1 * (sqrt(last_blocks)-1) + last_block_tau1;
        uparent = splitSpace([X(1,1) X(1,2)], [X(toIdx,1) X(toIdx,2)], tau2)';
        last_block_tau1 = cur_tau1 * (sqrt(last_blocks)-1) + last_block_tau1;
    end

%     fprintf(['h : ' num2str(h) 'uparent : ' mat2str(uparent) '\n']);
    for tau3_index = 1:last_blocks
        if size(X, 2) == 1 % 1d
            fromIdx = 1 + cur_tau1 * (tau3_index - 1);
            toIdx = cur_tau1 * tau3_index;
            cur_u = splitSpace(X(fromIdx), X(toIdx), tau2)';
        elseif size(X, 2) == 2 % 2d
            % if uparent =
            % [ a b ;
            %   c d ]
            % , then cur_u = a, b, c, or d.
            row = mod(tau3_index-1, tau3) + 1;
            col = floor((tau3_index-1) / tau3) + 1;
            fromIdx = 1 + [cur_tau1 * (row-1) cur_tau1 * (col-1)];
            toIdx = [cur_tau1 * row cur_tau1 * col];
            cur_u = splitSpace([X(fromIdx, 1) X(fromIdx, 2)], [X(toIdx, 1) X(toIdx, 2)], tau2)';
        end

        Kupup = feval(covfunc{:}, hyp(h+2,:), uparent) + tiny*eye(size_u);
        Kuup  = feval(cross_covfunc{:}, hyp(h+1,:), hyp(h+2,:), cur_u, uparent);
        Kuu  = feval(covfunc{:}, hyp(h+1,:), cur_u) + tiny*eye(size_u);
        L1 = chol(Kupup);
        Kuup_invL1 = Kuup/L1;
        
        Atau_last{tau3_index} = Kuup_invL1/L1';
        Qtau_last{tau3_index} = Kuu - Kuup_invL1*Kuup_invL1'+ tiny*eye(size_u);
    end
    if size(X, 2) == 1 % 1d
        pIndex2{h} = ceil((1:norm_blks)/tau3);
        if sum(pIndex2{h} == pIndex2{h}(end)) < tau3 && pIndex2{h}(end) > 1
            pIndex2{h}(pIndex2{h} == pIndex2{h}(end)) = pIndex2{h}(end) - 1;
        end
    elseif size(X, 2) == 2 % 2d
        parent_num_blks = norm_blks / group_factor;
        idxMatrix = kron(reshape(1:parent_num_blks, sqrt(parent_num_blks), sqrt(parent_num_blks)), ones(tau3));
        pIndex2{h} = idxMatrix(:);
    end
    norm_blks = floor(norm_blks / group_factor);
    
    At{h} = [repmat(Atau,norm_blks-1,1);Atau_last];
    Qt{h} = [repmat(Qtau,norm_blks-1,1);Qtau_last];
    cur_tau1 = cur_tau1 * tau3;
end
pIndex = {{}; pIndex2};

if size(X, 2) == 1 %1d
    for t = 1:T
        mInd = missingInd(t,:);
        sigma2 = vary*ones(size_s,1);
        sigma2(mInd) = sigma2(mInd)+sigmaInf; % set large variance for missing indices
        Rt{1}{t} = R + diag(sigma2);
    end
elseif size(X, 2) == 2 %2d
    for t = 1:T
        mInd = missingInd{t};
        sigma2 = vary*ones(size_s,1);
        sigma2(mInd(:)) = sigma2(mInd(:)) + sigmaInf; % set large variance for missing indices
        Rt{1}{t} = R + diag(sigma2);
    end
end

if nargout>5
    noVar = noHypSet*alpha+1; 
    dC = cell(noVar,1);
    dR = cell(noVar,1);
    for j = 1:noVar
        dC{j} = zeros(size(C));
        dR{j} = zeros(size(R));
        if j == noVar
            dR{j} = vary*eye(size(R));
        elseif j <= alpha
            Kuu = feval(covfunc{:},hyp(2,:),u) + tiny*eye(size_u);
            Ksudi = feval(cross_covfunc{:},hyp(1,:),hyp(2,:),s,u,1,mod(j-1,alpha)+1);
            Ksu = feval(cross_covfunc{:},hyp(1,:),hyp(2,:),s,u);
            Kssdi = feval(covfunc{:},hyp(1,:),s,[],mod(j-1,alpha)+1);
            Kb = Ksudi/Kuu;
            dC{j} = Kb;
            dR{j} = Kssdi -Kb*Ksu' - C*Ksudi';
        elseif j <= 2*alpha
            Kuu = feval(covfunc{:},hyp(2,:),u) + tiny*eye(size_u);
            Ksudi = feval(cross_covfunc{:},hyp(1,:),hyp(2,:),s,u,2,mod(j-1,alpha)+1);
            Ksu = feval(cross_covfunc{:},hyp(1,:),hyp(2,:),s,u);
            Kuudi = feval(covfunc{:},hyp(2,:),u,[],mod(j-1,alpha)+1);
            Ka = Kuudi/Kuu;
            Kb = Ksudi/Kuu;
            dC{j} = Kb - C*Ka;
            dR{j} = C*Ka*Ksu'-Kb*Ksu' - C*Ksudi';
        end
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
    last_block_tau1 = cur_tau1;
    norm_blks = noBlks;
    for h = 1:H-1

        blk_group_no = floor(norm_blks / group_factor) - 1;
        
        if size(X, 2) == 1 % 1d
            toIdx = min(cur_tau1 * tau3, X(end));
            uparent = splitSpace(X(1), X(toIdx), tau2)';
        elseif size(X, 2) == 2 % 2d
            toIdx = min(cur_tau1 * tau3, X(end, 1));
            uparent = splitSpace([X(1,1) X(1,2)], [X(toIdx,1) X(toIdx,2)], tau2)';
        end
        
        if norm_blks > group_factor
            last_blocks = group_factor + mod(norm_blks, group_factor);
        else
            last_blocks = norm_blks;
        end

        if size(X, 2) == 1 % 1d
            toIdx = cur_tau1 * (last_blocks-1) + last_block_tau1;
            uparent_l = splitSpace(X(1),X(toIdx),tau2)';
            last_block_tau1 = cur_tau1 * (last_blocks-1) + last_block_tau1;
        elseif size(X, 2) == 2 % 2d
            toIdx = cur_tau1 * (sqrt(last_blocks)-1) + last_block_tau1;
            uparent_l = splitSpace([X(1,1) X(1,2)], [X(toIdx,1) X(toIdx,2)], tau2)';
            last_block_tau1 = cur_tau1 * (sqrt(last_blocks)-1) + last_block_tau1;
        end

        %fprintf(['h : ' num2str(h) 'uparent_l : ' mat2str(uparent_l) '\n']);
        for j = 1:noVar
            if j > (2 + h) * alpha || j <= h * alpha % ignore hyperparameters above parent and below children
                dAt{h,j} = repmat({zeros(size(At{h}{1}))},length(At{h}),1);
                dQt{h,j} = repmat({zeros(size(Qt{h}{1}))},length(Qt{h}),1);
            else
                Atau = cell(group_factor,1);
                Qtau = cell(group_factor,1);
                for tau3_index = 1:min(group_factor, floor(X(end) / cur_tau1))
                    ori_At = At{h}{tau3_index};
                    if size(X, 2) == 1 % 1d
                        fromIdx = 1 + cur_tau1 * (tau3_index - 1);
                        toIdx = cur_tau1 * tau3_index;
                        cur_u = splitSpace(X(fromIdx), X(toIdx), tau2)';
                    elseif size(X, 2) == 2 % 2d
                        % if uparent =
                        % [ a b ;
                        %   c d ]
                        % , then cur_u = a, b, c, or d.
                        row = mod(tau3_index-1, tau3) + 1;
                        col = floor((tau3_index-1) / tau3) + 1;
                        fromIdx = 1 + [cur_tau1 * (row-1) cur_tau1 * (col-1)];
                        toIdx = [cur_tau1 * row cur_tau1 * col];
                        cur_u = splitSpace([X(fromIdx, 1) X(fromIdx, 2)], [X(toIdx, 1) X(toIdx, 2)], tau2)';
                    end
                    if j <= (h + 1) * alpha % consider my level
                        Kupup = feval(covfunc{:}, hyp(h+2,:), uparent) + tiny*eye(size_u);
                        Kuupdi  = feval(cross_covfunc{:}, hyp(h+1,:), hyp(h+2,:), cur_u, uparent,1,mod(j-1,alpha)+1);
                        Kuup  = feval(cross_covfunc{:}, hyp(h+1,:), hyp(h+2,:), cur_u, uparent);
                        Kuudi = feval(covfunc{:}, hyp(h+1,:),cur_u,[],mod(j-1,alpha)+1);
                        Kb = Kuupdi/Kupup;
                        Atau{tau3_index} = Kb;
                        Qtau{tau3_index} = Kuudi - Kb*Kuup' - ori_At*Kuupdi';
                    else % consider parent level
                        Kupup = feval(covfunc{:}, hyp(h+2,:), uparent) + tiny*eye(size_u);
                        Kuupdi  = feval(cross_covfunc{:}, hyp(h+1,:), hyp(h+2,:), cur_u, uparent,2,mod(j-1,alpha)+1);
                        Kuup  = feval(cross_covfunc{:}, hyp(h+1,:), hyp(h+2,:), cur_u, uparent);
                        Kupupdi = feval(covfunc{:}, hyp(h+2,:), uparent,[],mod(j-1,alpha)+1);
                        Ka = Kupupdi/Kupup;
                        Kb = Kuupdi/Kupup;
                        Atau{tau3_index} =  Kb - ori_At*Ka;
                        Qtau{tau3_index} =  ori_At*Ka*Kuup'- Kb*Kuup' - ori_At*Kuupdi';
                    end
                end
                Atau_last = cell(last_blocks,1);
                Qtau_last = cell(last_blocks,1);
                for tau3_index = 1:last_blocks
                    ori_At = At{h}{end-last_blocks+tau3_index};
                    if size(X, 2) == 1 % 1d
                        fromIdx = 1 + cur_tau1 * (tau3_index - 1);
                        toIdx = cur_tau1 * tau3_index;
                        cur_u = splitSpace(X(fromIdx), X(toIdx), tau2)';
                    elseif size(X, 2) == 2 % 2d
                        % if uparent =
                        % [ a b ;
                        %   c d ]
                        % , then cur_u = a, b, c, or d.
                        row = mod(tau3_index-1, tau3) + 1;
                        col = floor((tau3_index-1) / tau3) + 1;
                        fromIdx = 1 + [cur_tau1 * (row-1) cur_tau1 * (col-1)];
                        toIdx = [cur_tau1 * row cur_tau1 * col];
                        cur_u = splitSpace([X(fromIdx, 1) X(fromIdx, 2)], [X(toIdx, 1) X(toIdx, 2)], tau2)';
                    end
                    if j <= (h + 1) * alpha
                        Kupup = feval(covfunc{:}, hyp(h+2,:), uparent_l) + tiny*eye(size_u);
                        Kuupdi  = feval(cross_covfunc{:}, hyp(h+1,:), hyp(h+2,:), cur_u, uparent_l,1,mod(j-1,alpha)+1);
                        Kuup  = feval(cross_covfunc{:}, hyp(h+1,:), hyp(h+2,:), cur_u, uparent_l);
                        Kuudi = feval(covfunc{:}, hyp(h+1,:),cur_u,[],mod(j-1,alpha)+1);
                        Kb = Kuupdi/Kupup;
                        Atau_last{tau3_index} = Kb;
                        Qtau_last{tau3_index} = Kuudi - Kb*Kuup' - ori_At*Kuupdi';
                    else
                        Kupup = feval(covfunc{:}, hyp(h+2,:), uparent_l) + tiny*eye(size_u);
                        Kuupdi  = feval(cross_covfunc{:}, hyp(h+1,:), hyp(h+2,:), cur_u, uparent_l,2,mod(j-1,alpha)+1);
                        Kuup  = feval(cross_covfunc{:}, hyp(h+1,:), hyp(h+2,:), cur_u, uparent_l);
                        Kupupdi = feval(covfunc{:}, hyp(h+2,:), uparent_l,[],mod(j-1,alpha)+1);
                        Ka = Kupupdi/Kupup;
                        Kb = Kuupdi/Kupup;
                        Atau_last{tau3_index} =  Kb - ori_At*Ka;
                        Qtau_last{tau3_index} =  ori_At*Ka*Kuup'- Kb*Kuup' - ori_At*Kuupdi';
                    end
                end
                dAt{h,j} = [repmat(Atau,blk_group_no,1);Atau_last];
                dQt{h,j} = [repmat(Qtau,blk_group_no,1);Qtau_last];
            end
        end

        norm_blks = floor(norm_blks / group_factor);
        cur_tau1 = cur_tau1 * tau3;
    end


end

end
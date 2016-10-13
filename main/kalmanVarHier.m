
function  varargout = ...
    kalmanVarHier(At,Ct,Qt,Rt,pIndex,Kg,Yt,varargin)

%%
% Implements Kalman Smoothing or Kalman Filtering. Optionally returns
% the sufficient statistics of the Gaussian LDS. Based on Zoubin's
% code. Modified by Richard Turner. Further modified by Thang Bui
%
% x_{t}|x_{t-1} ~ Norm(A_{t} x_{t-1},Q_{t})
% y_{t}|x_{t} ~ Norm(C_{t} x_{t},R_{t})
% x_1 ~ Norm(x0,P0)
%
% With optional outputs and inputs:
%
% function
% [lik,Xfint,Pfint,Ptsum,YX,A1,A2,A3]=kalman(A,C,Q,R,x0,P0,Y,verbose,KF);
%
% see test_kalman.m for unit tests.
%
% INPUTS:
% At = cell of Dynamical Matrices, size {T} * [Kt,Kt]
% Ct = cell of Emission Matrices, size  {T} * [Dt,Kt]
% Qt = cell of State innovations noise, size {T} * [Kt,Kt]}
% Rt = cell of Emission Noise, size {T} * [Dt,Dt]
% x0 = initial state mean, size [K0,1]
% P0 = initial state covariance, size [K0,K0]
% Yt = Data, cell, size {T} * [Dt]
%
% OPTIONAL INPUTS:
% verbose = binary scalar, if set to 1 displays progress
%           information
% KF = binary scalar, if set to 1 carries out Kalman Filtering
%      rather than Kalman smoothing. Cannot return the sufficient
%      statistics in this case i.e. Ptsum, YX, A1, A2 and A3.
%
% OUTPUTS
% lik = likelihood
% Xfint = cell of posterior means, size {T} of  [Kt]
% Pfin = posterior covariance, size {T} of [Kt,Kt]
%
% OPTIONAL OUTPUTS:
% A4 = <x_{k t} x_{k' t-1}>, size {T} * [Kt,Kt] -- Added by Thang Bui


H = length(At);
T = length(Yt);

Acur=cell(H,1);
bcur=cell(H,1);
Pcur=cell(H,1);

Xfint=cell(H,1);   % P(x_t | y_1 ... y_T)    given all outputs
Pfint=cell(H,1);
CPfint = cell(H,1);

tiny = 1e-6;
QtInv = cell(size(Qt));
RtInv = cell(size(Rt{1}));

%%%%%%%%%%%%%%%
% UPWARD PASS: calculation of p(f_i|f_p,y)

logdetR = 0;
CRC = cell(H,1);
CRy = cell(H,1);
yRy = cell(H,1);
CRC{1} = cell(T,1);
CRy{1} = cell(T,1);
yRy{1} = cell(T,1);
for t=1:T
    LRt = chol(Rt{1}{t}+ tiny * eye(size(Rt{1}{t})) );
    RtInv{t} = LRt\(LRt'\eye(size(Rt{1}{t})));
    CL = LRt'\Ct{1}{t};
    YL = LRt'\Yt{t};
    CRC{1}{t} = CL' * CL;
    CRy{1}{t} = CL' * YL;
    yRy{1}{t} = YL' * YL;
    logdetR = logdetR + sum(log(diag(LRt)));
end

for h=1:H-1
    nNodes = length(At{h});
    Acur{h} = cell(nNodes,1);
    bcur{h} = cell(nNodes,1);
    Pcur{h} = cell(nNodes,1);
    QtInv{h} = cell(nNodes,1);
    for i=1:nNodes
        LQt = chol(Qt{h}{i}+ tiny * eye(size(Qt{h}{i})));
        QtInv{h}{i} = LQt\(LQt'\eye(size(Qt{h}{i})));
        Pcur{h}{i} = QtInv{h}{i} + CRC{h}{i};
        try
            LP = chol(Pcur{h}{i}+ tiny * eye(size(Pcur{h}{i})));
            logdetR = logdetR + sum(log(diag(LP))) + sum(log(diag(LQt)));
            Pcur{h}{i} = LP\(LP'\eye(size(Pcur{h}{i})));
        catch
            warning('error occured while chol...');
            PcurhiTmp = Pcur{h}{i}+ tiny * eye(size(Pcur{h}{i}));
            logdetR = logdetR + sum(log(eig(PcurhiTmp)))/2 + sum(log(diag(LQt)));
            Pcur{h}{i} = pinv(PcurhiTmp);
        end
        Acur{h}{i} = Pcur{h}{i} * (QtInv{h}{i} * At{h}{i});
        bcur{h}{i} = Pcur{h}{i} * CRy{h}{i};
    end
    npNodes = max(length(At{h+1}),1);
    CRC{h+1} = cell(npNodes,1);
    CRy{h+1} = cell(npNodes,1);
    yRy{h+1} = cell(npNodes,1);
    for i=1:npNodes
        CRC{h+1}{i} = 0;
        CRy{h+1}{i} = 0;
        yRy{h+1}{i} = 0;
    end
    for i=1:nNodes
        pId = pIndex{2}{h}(i);
        ACRC = At{h}{i}'*CRC{h}{i};
        ACRCS = ACRC*Pcur{h}{i};
        CRC{h+1}{pId} = CRC{h+1}{pId} + ACRC*At{h}{i} - ACRCS*ACRC';
        CRy{h+1}{pId} = CRy{h+1}{pId} + At{h}{i}'*CRy{h}{i} - ACRCS * CRy{h}{i};
        yRy{h+1}{pId} = yRy{h+1}{pId} + yRy{h}{i} - CRy{h}{i}' * Pcur{h}{i} * CRy{h}{i};
    end
end
%%%%%%%%%%%%%%%
% calculation of p(f_r|y)
try
    Kginv = eye(size(Kg))/Kg;
    Pfint{H} = {eye(size((Kginv+CRC{H}{1})))/(Kginv+CRC{H}{1})};
    Xfint{H} = {Pfint{H}{1}*CRy{H}{1}};
    P = Kginv+CRC{H}{1}+ tiny * eye(size(Kginv));
    cholP = chol(P);
    halfLogdetP = sum(log(diag(cholP)));
catch
    warning('error occured while chol');
    halfLogdetP = sum(log(eig(P))) / 2;
end

%%%%%%%%%%%%%%%
% for debugging:

% for h=1:H-1
%     nNodes = length(At{h});
%     npNodes = max(length(At{h+1}),1);
%     Ct{h+1} = cell(npNodes,1);
%     Rt{h+1} = cell(npNodes,1);
%     for i=1:nNodes
%         pId = pIndex{2}{h}(i);
%         Ct{h+1}{pId} = [Ct{h+1}{pId}; Ct{h}{i} * At{h}{i}];
%         RCQC = Rt{h}{i} + Ct{h}{i} * Qt{h}{i} * Ct{h}{i}';
%         Rt{h+1}{pId} = blkdiag(Rt{h+1}{pId}, RCQC);
%     end
% end
% yvar = Rt{H}{1} + Ct{H}{1} * Kg * Ct{H}{1}';
% dd = chol(yvar);
% detef = -sum(log(diag(dd)))
% ytot = cell2mat(Yt');
% fianl = -ytot'*(yvar\ytot)/2.0

%%%%%%%%%%%%%%%
% calculation of likelihood

lik1 = -T*length(Yt{1})/2*log(2*pi);
lik2 = -halfLogdetP;
lik3 = -log(det(Kg))/2;
lik4 = -logdetR;
lik5 = -yRy{H}{1}/2;
lik6 = +CRy{H}{1}' * Pfint{H}{1} * CRy{H}{1}/2;


lik = lik1 + lik2 + lik3 + lik4 + lik5 + lik6;
%%%%%%%%%%%%%%%
% DOWNWARD PASS

for h=(H-1):-1:1
    nNodes = length(At{h});
    Xfint{h} = cell(nNodes,1);
    Pfint{h} = cell(nNodes,1);
    for i=1:nNodes
        pId = pIndex{2}{h}(i);
        Xfint{h}{i} = Acur{h}{i} * Xfint{h+1}{pId} + bcur{h}{i};
        CPfint{h}{i} = Acur{h}{i} * Pfint{h+1}{pId};
        Pfint{h}{i} = Pcur{h}{i} + CPfint{h}{i} * Acur{h}{i}';
    end
end


%% FIND DERIVATIVES
if length(varargin) >= 1
    dAt = varargin{1};
    dCt = varargin{2};
    dQt = varargin{3};
    dRt = varargin{4};
    dKg = varargin{5};
    noVar = size(dAt,2);
    
    dlik = zeros(noVar,1);
    for j = 1:noVar
        M15 = -1/2*sum(sum(Kginv.*dKg{j}'));
        M16 = -Kginv*dKg{j}*Kginv;
        dlik(j) = dlik(j) + M15 - 1/2*(sum(sum(M16.*Pfint{H}{1}')) + Xfint{H}{1}'*M16*Xfint{H}{1});
    end
    for h=(H-1):-1:1
        nNodes = length(At{h});
        for i=1:nNodes
            pId = pIndex{2}{h}(i);
            mu1 = Xfint{h}{i};
            sig11 = Pfint{h}{i};
            mu2 = Xfint{h+1}{pId};
            sig22 = Pfint{h+1}{pId};
            sig12 = CPfint{h}{i};
            Qinv = QtInv{h}{i};
            
            M11 = sig12/sig22;
            
            for j = 1:noVar
                M5 = -1/2*sum(sum(Qinv.*dQt{h,j}{i}'));
                M6 = -Qinv*dQt{h,j}{i}*Qinv;
                M7 = Qinv*dAt{h,j}{i} + M6*At{h}{i};
                M8 = dAt{h,j}{i}'*(Qinv*At{h}{i}) + At{h}{i}'*M7;

                Lb2 = -1/2*(sum(sum(M6.*sig11')) + mu1'*M6*mu1);
                Lb4 = -1/2*(sum(sum(M8.*sig22')) + mu2'*M8*mu2);
                Lb3 = mu1'*M7*mu2 + sum(sum((M11'*M7).*sig22'));
                dlik(j) = dlik(j) + M5 + Lb2 + Lb3 + Lb4;
            end
        end
    end
    for i=1:T
        for j = 1:noVar
            mu1 = Xfint{1}{i};
            sig11 = Pfint{1}{i};

            M0 = RtInv{i}*dRt{i,j};
            M1 = -1/2*trace(M0);
            M2 = -M0*RtInv{i};
            M3 = RtInv{i}*dCt{i,j} + M2*Ct{1}{i};
            M4 = dCt{i,j}'*RtInv{i}*Ct{1}{i} + Ct{1}{i}'*M3;
            
            dlik(j) = dlik(j) + M1 - 1/2*Yt{i}'*M2*Yt{i} + Yt{i}'*M3*mu1 ...
                      - 1/2*(mu1'*M4*mu1 + sum(sum(M4.*sig11')));
        end
    end
    
end


if length(varargin) >= 3
    varargout{1} = lik;
    varargout{2} = dlik;
else
    varargout{1} = Xfint;
    varargout{2} = Pfint;
    varargout{3} = CPfint;
end
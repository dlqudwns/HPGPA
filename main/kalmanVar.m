% function  [lik,Xfint,Pfint,A4,dlik] = ...
%     kalmanVar(At,Ct,Qt,Rt,x0,P0,Yt,varargin)

function  varargout = ...
    kalmanVar(At,Ct,Qt,Rt,x0,P0,Yt,varargin)


% function
% [lik,Xfin,Pfin]=kalman(A,C,Q,R,x0,P0,Y);
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

T = length(Yt);

problem=0;
lik=0;

Xcur=cell(T,1);   % P(x_t | y_1 ... y_t)
Xfint=cell(T,1);   % P(x_t | y_1 ... y_T)    given all outputs

Ppre=cell(T,1);
Pcur=cell(T,1);
Pfint=cell(T,1);
J = cell(T,1);

if nargin<=7
    verbose = 0 ;
else
    verbose = varargin{1};
end
if nargin<=8
    KF = 0 ;
else
    KF = varargin{2};
end

if verbose==1
    if KF==1
        disp('Kalman Filtering')
    else
        disp('Kalman Smoothing')
    end
end

%%%%%%%%%%%%%%%
% FORWARD PASS

Xpre=x0;

P0tin = P0 + 1e-7*eye(length(P0));
Ppre{1} = P0tin;

CntInt=T/5; % T / 2*number of progress values displayed

for t=1:T
    Rcur = Rt{t};
    Ccur = Ct{t};
    Kt = size(Ccur,2);
    Dt = size(Rcur,1);
    ItK = eye(Kt);
    ItD = eye(Dt);
    Ppret = Ppre{t};
    
    if verbose==1&&mod(t-1,CntInt)==0
        fprintf(['Progress ',num2str(floor(50*t/T)),'%%','\r'])
    end
    
    temp1=Rcur+Ccur*Ppret*Ccur';
    invP=ItD/temp1;
    %CP=Ccur'/temp1;
    CP=Ccur'*invP;
    Kcur=Ppret*CP; % Kalman gain
    KC=Kcur*Ccur;
    Ydiff=Yt{t}-Ccur*Xpre;
    Xcur{t}=Xpre+Kcur*Ydiff;
    %Pcur{t}=Ppret-KC*Ppret;
    
    % numerical problem with subtraction, use Joseph form
    % for the covariance update
    IsubKC = ItK-KC;
    Pcur{t} = IsubKC*Ppret*IsubKC' + Kcur*Rcur*Kcur';
    
    if (t<T)
        Anext = At{t+1};
        Qnext = Qt{t+1};
        Xpre=Anext*Xcur{t};
        Ppre{t+1}=Anext*Pcur{t}*Anext'+Qnext;
    end
    
    % calculate likelihood
    %[cholinvP,e1] = chol(invP);
    %if e1
    %disp('Problem!!, switching to chol(P)')
    if length(varargin) >= 3
        P = Rcur + Ccur*Ppret*Ccur';
        [cholP,e2] = chol(P);
        if e2
            problem = 1;
            %keyboard
            continue;
        else
            logdetP=sum(log(diag(cholP)));
            lik=lik-Dt/2*log(2*pi)-logdetP-0.5*sum(sum(Ydiff.*(invP*Ydiff)));
        end
    end
        %else
        %logdetiP=sum(log(diag(cholinvP)));
        %lik=lik+logdetiP-Dt/2*log(2*pi)-0.5*sum(sum(Ydiff.*(invP*Ydiff)));
        %end
end

% Kalman Filtering

if KF==1
    % Only Kalman filtering
    Xfint=Xcur;
    Pfint=Pcur;
else
    % Kalman Smoothing
    
    %%%%%%%%%%%%%%%
    % BACKWARD PASS
    
    t=T;
    Xfint{t}=Xcur{t};
    Pfint{t}=Pcur{t};
    
    for t=(T-1):-1:1
        Anext = At{t+1};
        if verbose==1&&mod(t+1,CntInt)==0
            fprintf(['Progress ',num2str(50+floor(50*(T-t+1)/T)),'%%','\r'])
        end
        J{t}=Pcur{t}*Anext'/Ppre{t+1};
        Xfint{t}=Xcur{t}+J{t}*(Xfint{t+1}-Anext*Xcur{t});
        Pfint{t}=Pcur{t}+J{t}*(Pfint{t+1}-Ppre{t+1})*J{t}';
    end
end


A4=cell(T-1,1);
%Pcov=(eye(Kt)-KC)*At{T}*Pcur{T-1};
Pcov = Pfint{T}*J{T-1}';
A4{T-1}=Pcov;
for t=(T-1):-1:2
    %Pcov=(Pcur{t}+J{t}*(Pcov-At{t+1}*Pcur{t}))*J{t-1}';
    Pcov = Pfint{t}*J{t-1}';
    A4{t-1} = Pcov;
    
end

%keyboard
%% FIND DERIVATIVES
if length(varargin) >= 3
    dAt = varargin{3};
    dCt = varargin{4};
    dQt = varargin{5};
    dRt = varargin{6};
    dP0 = varargin{7};
    noVar = size(dAt,2);
    dlik = zeros(noVar,1);
    
    for i = 1:T
        mu1 = Xfint{i};
        sig11 = Pfint{i};
        
        if i>1
            mu2 = Xfint{i-1};
            sig22 = Pfint{i-1};
            sig12 = A4{i-1};
            M11 = sig12/sig22;
            M11mu2 = M11*mu2;
            M10 = mu1 - M11*mu2;
        end
        Qtinv = Qt{i}\eye(size(Qt{i}));
        Rtinv = Rt{i}\eye(size(Rt{i}));
        for j = 1:noVar
            if i==1
                P0inv = P0tin\eye(size(P0tin));
                %M15 = -1/2*trace(P0inv*dP0{j});
                M15 = -1/2*sum(sum(P0inv.*dP0{j}'));
                M16 = -P0inv*dP0{j}*P0inv;
                %dlik(j) = dlik(j) + M15 - 1/2*(trace(M16*sig11) + mu1'*M16*mu1);
                dlik(j) = dlik(j) + M15 - 1/2*(sum(sum(M16.*sig11')) + mu1'*M16*mu1);
            else
                M5 = -1/2*trace(Qtinv*dQt{i,j});
                M6 = -Qtinv*dQt{i,j}*Qtinv;
                M7 = Qtinv*dAt{i,j} + M6*At{i};
                M8 = dAt{i,j}'*(Qtinv*At{i}) + At{i}'*M7;
                
%                 Lb2 = -1/2*(trace(M6*sig11) + mu1'*M6*mu1);
%                 Lb4 = -1/2*(trace(M8*sig22) + mu2'*M8*mu2);
%                 Lb3 = M10'*M7*mu2 + trace(M11'*M7*sig22) + M11mu2'* ...
%                     M7*mu2;
                Lb2 = -1/2*(sum(sum(M6.*sig11')) + mu1'*M6*mu1);
                Lb4 = -1/2*(sum(sum(M8.*sig22')) + mu2'*M8*mu2);
                Lb3 = M10'*M7*mu2 + sum(sum((M11'*M7).*sig22')) + M11mu2'* ...
                    M7*mu2;
                dlik(j) = dlik(j) + M5 + Lb2 + Lb3 + Lb4;
            end
            

            M0 = Rtinv*dRt{i,j};
            M1 = -1/2*trace(M0);
            M2 = -M0*Rtinv;
            %M1 = -1/2*trace(Rtinv*dRt{i,j});
            %M2 = -Rtinv*dRt{i,j}*Rtinv;
            M3 = Rtinv*dCt{i,j} + M2*Ct{i};
            M4 = dCt{i,j}'*Rtinv*Ct{i} + Ct{i}'*M3;
            
%             dlik(j) = dlik(j) + M1 - 1/2*Yt{i}'*M2*Yt{i} + Yt{i}'*M3*mu1 ...
%                       - 1/2*(mu1'*M4*mu1 + trace(M4*sig11));
                  
            dlik(j) = dlik(j) + M1 - 1/2*Yt{i}'*M2*Yt{i} + Yt{i}'*M3*mu1 ...
                      - 1/2*(mu1'*M4*mu1 + sum(sum(M4.*sig11')));
            
        end
    end
end


if problem
    fprintf('!!!!!!!!! problem  !!!!!!!!!\r\n');
    %lik = NaN;
end

if verbose==1
    fprintf('                                        \r')
end

if length(varargin) >= 3
    varargout{1} = lik;
    varargout{2} = dlik;
else
    varargout{1} = Xfint;
    varargout{2} = Pfint;
    varargout{3} = A4;
end
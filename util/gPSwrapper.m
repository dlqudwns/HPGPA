function [ f, df ] = gPSwrapper(thetaj,j,theta,covfunc,X,tau1,tau2,noBlks,missingInd)
%GPSHWRAPPER 이 함수의 요약 설명 위치
%   자세한 설명 위치

theta(j) = thetaj;

[Ct,Rt,At,Qt,indexer,dCt,dRt,dAt,dQt] = getParamsSETree(covfunc,theta(1:end-1),exp(theta(end)),X,tau1,tau2,noBlks,missingInd);
%[Ct,Rt,At,Qt,dCt,dRt,dAt,dQt] = getParamsSE(covfunc,theta(1:end-1),exp(theta(end)),X,tau1,tau2,noBlks,missingInd);



f = 0;
df = 0;
for h=3:length(Qt)
    for index=1:length(Qt{h})
        f = f + sum(sum(Rt{h}{index}));
        df = df + sum(sum(dRt{h}{index,j}));
    %     f = f + sum(sum(Rt{h}{index}));
    %     df = df + sum(sum(dRt{index,j}));
    end
end

%f = sum(sum(Rt{1}));
%df = sum(sum(dRt{1,j}));


%% to test hypes
% hyp1 = theta_init(1:2);
% hyp2 = theta_init(3:4);
% K = crosscovSEiso(hyp1, hyp2, X);
% 
% d1 = sum(sum(crosscovSEiso(hyp1, hyp2, X, [], 1, 1)));
% d2 = sum(sum(crosscovSEiso(hyp1, hyp2, X, [], 1, 2)));
% d3 = sum(sum(crosscovSEiso(hyp1, hyp2, X, [], 2, 1)));
% d4 = sum(sum(crosscovSEiso(hyp1, hyp2, X, [], 2, 2)));
% 
% f = sum(sum(K));
% df = [d1, d2, d3, d4]';

end

  
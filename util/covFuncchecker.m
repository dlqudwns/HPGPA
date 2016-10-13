function [ f, df ] = covFuncchecker( theta_init, X )
%PARAMSGRADCHECKER 이 함수의 요약 설명 위치
%   자세한 설명 위치

hyp1 = theta_init(1:2);
hyp2 = theta_init(3:4);
K = crosscovSEiso(hyp1, hyp2, X);

d1 = sum(sum(crosscovSEiso(hyp1, hyp2, X, [], 1, 1)));
d2 = sum(sum(crosscovSEiso(hyp1, hyp2, X, [], 1, 2)));
d3 = sum(sum(crosscovSEiso(hyp1, hyp2, X, [], 2, 1)));
d4 = sum(sum(crosscovSEiso(hyp1, hyp2, X, [], 2, 2)));

f = sum(sum(K));
df = [d1, d2, d3, d4]';

end


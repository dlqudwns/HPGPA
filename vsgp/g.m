function [ output ] = g( x, y, sigma )
%G 이 함수의 요약 설명 위치
%   자세한 설명 위치
    
    coeff = prod(2 * pi * sigma)^(-0.5);
    output = coeff * exp(-0.5 * sum(((x-y).^2)./sigma));
end


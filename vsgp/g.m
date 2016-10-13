function [ output ] = g( x, y, sigma )
%G �� �Լ��� ��� ���� ��ġ
%   �ڼ��� ���� ��ġ
    
    coeff = prod(2 * pi * sigma)^(-0.5);
    output = coeff * exp(-0.5 * sum(((x-y).^2)./sigma));
end


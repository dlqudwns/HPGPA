function [ H ] = computeH( numBlocks, tau3 )

H = 1;
tmp = numBlocks;
while tmp > 1
    H = H + 1;
    tmp = floor(tmp / tau3);
end

end


function [ theta ] = expand_theta( compressed_theta, H )

theta = [repmat(compressed_theta(1:2), H, 1); compressed_theta(3)]; % assume that compressed_theta is 3d.
for i=3:2:2*H
    theta(i) = theta(i) + log(2) * (floor(i/2) - 1);
end
theta(5:2:end-1) = theta(5:2:end-1) + 1;

end


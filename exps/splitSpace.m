function [ X ] = splitSpace( from, to, N )

if length(from) == 1 && length(to) == 1
    X = linspace(from-0.5, to+0.5, N);
elseif length(from) == 2 && length(to) == 2
    X = [repmat(linspace(from(1), to(1), N), 1, N) ; kron(linspace(from(2), to(2), N), ones(1, N))];
else
    error('splitSpace is not defined for the given size');
end

end

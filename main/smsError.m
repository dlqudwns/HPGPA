function err = smsError(ftrue,fest)
err = inf;
if isempty(ftrue) || isempty(fest) || length(ftrue)~=length(fest)
    error('Empty input vector or vectors of different length')
end


err = sum((fest(:)-ftrue(:)).^2)/length(fest);
err = err/var(ftrue);
end
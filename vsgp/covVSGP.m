function [K,Kuu,Ku] = covVSGP(xu, hyp, x, z, i)

if nargin<3, K = int2str(2+size(xu,1)); return, end
if nargin<4, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

ell = exp(2*hyp(1));                                 % characteristic length scale
sf2 = exp(2*hyp(2));                                           % signal variance
lengthscales = exp(2*hyp(3:end));

D = size(x,2);
if size(xu,2) ~= size(x,2)
  error('Dimensionality of inducing inputs must match training inputs');
end
if size(lengthscales,1) ~= size(xu,1)
    error('Number of inducing inputs must match number of lengthscales');
end

if dg
    K = zeros(size(x,1),1);
else
    if xeqz
        K = zeros(size(x,1),1);
        if nargout>1
            Kuu = sq_dist(xu')./(bsxfun(@plus, lengthscales, lengthscales')-ell);
        end
        if nargout>2
            Ku = bsxfun(@rdivide,sq_dist(xu',x'),lengthscales);
        end
    else
      K = bsxfun(@rdivide,sq_dist(xu',z'),lengthscales);
    end
end

if nargin<5                                                        % covariances
    if dg || xeqz
        K = sf2*exp(-K/2)*((2*pi*ell)^(-D/2));
    else
        K = bsxfun(@rdivide,exp(-K/2),((2*pi*lengthscales).^(D/2)));
    end
    if nargout>1
        ls = bsxfun(@plus, lengthscales, lengthscales')-ell;
        Kuu = (exp(-Kuu/2)./((2*pi*ls).^(D/2)))/sf2;
    end
    if nargout>2
        Ku = bsxfun(@rdivide,exp(-Ku/2),((2*pi*lengthscales).^(D/2)));
    end
else                                                               % derivatives
    if i==1
        if dg || xeqz
            K = - D *sf2*exp(-K/2)*((2*pi*ell)^(-D/2));
        else
            K = zeros(size(K));
        end
        if nargout>1
            ls = bsxfun(@plus, lengthscales, lengthscales')-ell;
            Kuu = D * (exp(-Kuu/2)./((2*pi*ls).^(D/2)))/sf2*ell./ls - (exp(-Kuu/2)./((2*pi*ls).^(D/2)))/sf2*ell./ls.*Kuu;
        end
        if nargout>2,  Ku = zeros(size(Ku));  end
    elseif i==2
        if dg || xeqz
            K = 2*sf2*exp(-K/2)*((2*pi*ell)^(-D/2));
        else
            K = zeros(size(K));
        end
        if nargout>1
            ls = bsxfun(@plus, lengthscales, lengthscales')-ell;
            Kuu = -2*(exp(-Kuu/2)./((2*pi*ls).^(D/2)))/sf2;
        end
        if nargout>2,  Ku = zeros(size(Ku));  end
    elseif i > 2 && i <= length(hyp)
        if dg || xeqz
            K = zeros(size(K));
        else
            K2 = bsxfun(@rdivide,exp(-K/2).*K,((2*pi*lengthscales).^(D/2))) - D * bsxfun(@rdivide,exp(-K/2),((2*pi*lengthscales).^(D/2)));
            K = zeros(size(K));
            K(i-2,:) = K2(i-2,:);
        end
        if nargout>1
            ls = bsxfun(@plus, lengthscales, lengthscales')-ell;
            K2 = bsxfun(@times,(exp(-Kuu/2)./((2*pi*ls).^(D/2)))/sf2./ls.*Kuu, lengthscales) - D * bsxfun(@times,(exp(-Kuu/2)./((2*pi*ls).^(D/2)))/sf2./ls, lengthscales);
            Kuu = zeros(size(Kuu)); 
            Kuu(i-2,:) = K2(i-2,:); 
            Kuu(:,i-2) = Kuu(:,i-2) + K2(i-2,:)'; 
        end
        if nargout>2
            K2 = bsxfun(@rdivide,exp(-Ku/2).*Ku,((2*pi*lengthscales).^(D/2))) - D * bsxfun(@rdivide,exp(-Ku/2),((2*pi*lengthscales).^(D/2)));
            Ku = zeros(size(Ku));
            Ku(i-2,:) = K2(i-2,:); 
        end
    else
        error('Unknown hyperparameter')
    end
end
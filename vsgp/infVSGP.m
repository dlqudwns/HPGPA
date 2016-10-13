function [post nlZ dnlZ] = infVSGP(hyp, mean, cov, lik, x, y)

if iscell(lik), likstr = lik{1}; else likstr = lik; end
if ~ischar(likstr), likstr = func2str(likstr); end
if ~strcmp(likstr,'likGauss')               % NOTE: no explicit call to likGauss
  error('Inference with inVSGP only possible with Gaussian likelihood.');
end
cov1 = cov{1}; if isa(cov1, 'function_handle'), cov1 = func2str(cov1); end
if ~strcmp(cov1,'covVSGP'); error('Only covVSGP supported.'), end    % check cov

[diagK,Kuu,Ku] = feval(cov{:}, hyp.cov, x);         % evaluate covariance matrix
m = feval(mean{:}, hyp.mean, x);                          % evaluate mean vector
[n, D] = size(x); nu = size(Kuu,1);
cov{2} = reshape(cov{2},[],D);

sn2  = exp(2*hyp.lik);                              % noise variance of likGauss
snu2 = 1e-6*sn2;                              % hard coded inducing inputs noise
Luu  = chol(Kuu+snu2*eye(nu));                         % Kuu + snu2*I = Luu'*Luu
V  = Luu'\Ku;                                     % V = inv(Luu')*Ku => V'*V = Q
g_sn2 = diagK + sn2 - sum(V.*V,1)';          % g + sn2 = diag(K) + sn2 - diag(Q)
Lu = chol(eye(nu) + (V./repmat(g_sn2',nu,1))*V');  % Lu'*Lu=I+V*diag(1/g_sn2)*V'
r  = (y-m)./sqrt(g_sn2);
be = Lu'\(V*(r./sqrt(g_sn2)));
iKuu = solve_chol(Luu,eye(nu));                       % inv(Kuu + snu2*I) = iKuu
post.alpha = Luu\(Lu\be);                      % return the posterior parameters
post.L  = solve_chol(Lu*Luu,eye(nu)) - iKuu;                    % Sigma-inv(Kuu)
post.sW = ones(n,1)/sqrt(sn2);           % unused for FITC prediction  with gp.m

if nargout>1                                % do we want the marginal likelihood
  nlZ = sum(log(diag(Lu))) + (sum(log(g_sn2)) + n*log(2*pi) + r'*r - be'*be)/2;
  if nargout>2                                         % do we want derivatives?
    dnlZ = hyp;                                 % allocate space for derivatives
    al = r./sqrt(g_sn2) - (V'*(Lu\be))./g_sn2;      % al = (Kt+sn2*eye(n))\(y-m)
    B = iKuu*Ku; w = B*al;
    W = Lu'\(V./repmat(g_sn2',nu,1));
    for i = 1:numel(hyp.cov)
      [ddiagKi,dKuui,dKui] = feval(cov{:}, hyp.cov, x, [], i);  % eval cov deriv
      R = 2*dKui-dKuui*B; v = ddiagKi - sum(R.*B,1)';   % diag part of cov deriv
      dnlZ.cov(i) = (ddiagKi'*(1./g_sn2)+w'*(dKuui*w-2*(dKui*al))-al'*(v.*al)...
                                 - sum(W.*W,1)*v - sum(sum((R*W').*(B*W'))) )/2;
    end
    diag_dK = 1./g_sn2 - sum(W.*W,1)' - al.*al;                  % diag(dnlZ/dK)
    dnlZ.lik = sn2*sum(diag_dK);
    % since snu2 is a fixed fraction of sn2, there is a covariance-like term in
    BWt = B*W'; % the derivative as well
    dKuui = 2*snu2; R = -dKuui*B; v = -sum(R.*B,1)';    % diag part of cov deriv
    dnlZ.lik = dnlZ.lik + (w'*dKuui*w -al'*(v.*al)...
                                    - sum(W.*W,1)*v - sum(sum((R*W').*BWt)) )/2; 
    for i = 1:numel(hyp.mean)
      dnlZ.mean(i) = -feval(mean{:}, hyp.mean, x, i)'*al;
    end
  end
end
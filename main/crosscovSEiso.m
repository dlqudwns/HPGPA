function K = crosscovSEiso(hyp1, hyp2, x, z, which, i)
%CROSSCOVSEISO 이 함수의 요약 설명 위치
%   자세한 설명 위치


if nargin<3, K = '4'; return; end                  % report number of parameters
if nargin<4, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

ell = sqrt(exp(2*hyp1(1))+exp(2*hyp2(1)));                                 % characteristic length scale
[n,D] = size(x);
numer = exp(D*(hyp1(1)+hyp2(1))/2);
denom = (ell)^(D);
sf2 = exp(hyp1(2)+hyp2(2));                                           % signal variance
const = 2^(D/2)*sf2*numer/denom;

% precompute squared distances   
if dg                                                               % vector kxx
  sqd = zeros(size(x,1),1);
else
  if xeqz                                                 % symmetric matrix Kxx
    sqd = sq_dist(x'/ell);
  else                                                   % cross covariances Kxz
    sqd = sq_dist(x'/ell,z'/ell);
  end
end

if nargin<5                                                        % covariances
  K = const*exp(-sqd);
else                                                               % derivatives
  if i==1
    if which==1
        frac = exp(2*hyp1(1))/(ell^2);
    elseif which==2
        frac = exp(2*hyp2(1))/(ell^2);
    else
      error('Unknown which')
    end
    K = const*exp(-sqd).*(D/2- D*frac+2*sqd*frac);
  elseif i==2
    K = const*exp(-sqd);
  else
    error('Unknown hyperparameter')
  end
end

end
  

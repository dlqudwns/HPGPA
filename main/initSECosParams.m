function params = initSECosParams(s,fs,K,wsize,ovlapPc,varargin)
% initSECosParams Initialise spectral mixture parameters

window = round(length(s)/wsize);
overlap = round(ovlapPc*window);
[pxx,f] = pwelch(s,window,overlap,[],fs,'onesided','power');
[pks,locs] = findpeaks(pxx,'SortStr','descend');

% the spectrum is over-smoothed, there may be less than K peaks
% randomly choose some more
if length(pks) < K
    remaining = setdiff(1:length(pxx),locs)';
    needed = K-length(pks);
    chosenIdx = randperm(length(remaining),needed);
    locs = [locs; remaining(chosenIdx)];
end
locs = locs(1:K);
fpeaks = f(locs);
pks = pxx(locs);
pks = pks/min(pks)+1;
% TODO: may need to get subband data and choose lengthscales accordingly

sigstd = std(s);
params = zeros(K*3+1,1);
params(1:2:2*K-1)   = 50;           % lengthscales
%params(2:2:2*K)     = 1/2*sigstd;   % sigma_signal
params(2:2:2*K)     = log(pks);
params(2*K+1:3*K)   = fpeaks/fs;    % frequencies
params(3*K+1)       = 1/4*sigstd;   % sigma_noise

if length(varargin) == 1
    opt = varargin{1};
    if opt == 1 % plot
        pxxlog = 10*log10(pxx);
        figure(1), plot(f/fs,pxxlog,'-b',fpeaks/fs,pxxlog(locs),'+r')
        xlabel('freq'), ylabel('power (dB)');
    end
end

end
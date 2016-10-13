function [theta_end,nlml] = trainSE(theta_init,covfunc,...
    Xtrain,Ytrain,tau1,tau2,missingInd,noEvals)
K = length(Xtrain)/tau1;
if size(Xtrain, 2) == 1 % 1d
    y1 = reshape(Ytrain,tau1,K);
    yKalman = num2cell(y1,1);
    adj_matrix = [];
elseif size(Xtrain, 2) == 2 % 2d
    yKalman = {};
    for j=1:K
        for i=1:K
            yKalman{end+1} = Ytrain((i-1)*tau1+1:i*tau1, (j-1)*tau1+1:j*tau1);
            yKalman{end} = yKalman{end}(:);
        end
    end
    noBlks = K^2;
    % Kruskal
    tic;
    coordinate = splitSpace([0 0], [1 1], K)';
    adj_matrix = ones(noBlks)-eye(noBlks);
    weight_matrix = (sq_dist(coordinate') + sq_dist(coordinate')')/2;
    adj_matrix = kruskal(adj_matrix, weight_matrix);
    fprintf('Kruskal time : %f\n', toc);
else
    error('dimension of Xtrain?')
end
tic;

fname = 'objFunctionSE';

% fprintf('Check grad gPSHwrapper\n');

% for j=1:length(theta_init)
%     checkgrad('gPSwrapper',theta_init(j),1e-4,j,theta_init,covfunc,...
%         Xtrain,tau1,tau2,K,missingInd);
% end

%test derivativevest(ind) = diag(C1*Pfint{i}*C1' + R1);
% d = checkgrad(fname,theta_init,1e-6,covfunc,...
%      Xtrain,yKalman,tau1,tau2,missingInd);
% % keyboard

opt.length = -noEvals;
opt.method = 'BFGS';
opt.verbosity = 0;
[theta_end,lik,i] = minimize_new(theta_init,fname,opt,covfunc,...
    Xtrain,yKalman,tau1,tau2,missingInd,adj_matrix);
nlml = lik(end);
end
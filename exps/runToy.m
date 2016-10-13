% function [] = runToy(iter, missing_length)
rng(1);

%% Parameters
% 1: treeGP
% 2: localGP
% 3: hierGP with single cov func
% 4: hierGP with multi cov func
% 5: FITC
% 6: FullGP
algorithm = 7;

iter = 1;
missing_length = 200;
noEvals = 200;
tau1 = 50; % the number of inputs poins per each block
tau2 = 5; % the number of inducing points per each block
tau3 = 2; % only used for hierarchical GP. block grouping factor
numInducingFITC = 300;
leaf_lengthscale = 120;
dataset = 'mydata2';

results = struct();
for it = 1:iter
fprintf([num2str(it) 'th iteration...'])
%% load data, setting
[y, yOri] = loadData(dataset);
T = length(y);

%% setting missing blocks
s = missing_length;
mInd = randi(T, 5, 1);
missingInd = zeros(T,1);
for m = 1:length(mInd);
    j = (mInd(m))+(1:s);
    j(j>T) = [];
    missingInd(j) = 1;
end
missingInd = missingInd==1;

%% 
YtrainOri = y;
YtrainOri(missingInd) = 0;
YtestOri = y;
YtestOri(missingInd) = 0;

%% parameters
K = floor(T/tau1); % total number of blocks
Xtrain = 1:tau1*K;
Ytrain = YtrainOri(Xtrain);
Ytest = YtestOri(Xtrain);
missing = missingInd(Xtrain);
missingStack = reshape(missing,[tau1,K])';

covfunc = {@covSEiso};
cross_covfunc = {@crosscovSEiso};

H = floor(log(K)/log(tau3))+2;
ind_params = eval(feval(covfunc{:})); % the number of independent free parameters. 2.

%% Set initial theta lengthscale, signal variance, noise variance
theta_init = log([leaf_lengthscale, sqrt(var(Ytrain))/2, sqrt(var(Ytrain))/2]');

if algorithm == 4 % for hierarchical GP with multiple covariance functions
    theta_init = expand_theta(theta_init, H);
end

%% train
tic;
if algorithm == 1
    [theta_end, ~] = trainSE(theta_init,covfunc,Xtrain',Ytrain,tau1,tau2,missingStack,noEvals);
elseif algorithm == 2
    [theta_end, ~] = trainSELocal(theta_init,covfunc,Xtrain',Ytrain,tau1,tau2,missingStack,noEvals);
elseif algorithm == 3
    [theta_end, ~] = trainSEHier_single(theta_init,covfunc,Xtrain',Ytrain,tau1,tau2,tau3,missingStack,noEvals);
elseif algorithm == 4
    [theta_end, ~] = trainSEHier(theta_init,covfunc,cross_covfunc, Xtrain',Ytrain,tau1,tau2,tau3,missingStack,noEvals,H);
elseif algorithm == 5
    [theta_end, ~] = trainSEFITC(theta_init,covfunc,Xtrain(~missing)',Ytrain(~missing),numInducingFITC,noEvals);
elseif algorithm == 6
    [theta_end, ~] = trainFullGP(theta_init,covfunc,Xtrain(~missing)',Ytrain(~missing),noEvals);
elseif algorithm == 7
    [theta_end, ~] = trainVSGP(theta_init,Xtrain(~missing)',Ytrain(~missing),numInducingFITC,noEvals);
else
    error('no algorithm');
end
results(it).trainTime = toc;
fprintf('Training time : %fs\n', results(it).trainTime);

%% predict
tic;
if algorithm == 1
    [fest,vest] = predictSE(theta_end,covfunc,Xtrain',Ytest,tau1,tau2,missingStack);
elseif algorithm == 2
    [fest,vest] = predictSELocal(theta_end,covfunc,Xtrain',Ytest,tau1,tau2,missingStack);
elseif algorithm == 3
    [fest,vest] = predictSEHier_single(theta_end,covfunc,Xtrain',Ytest,tau1,tau2,tau3,missingStack);
elseif algorithm == 4
    [fest,vest, Xfint, ~] = predictSEHier(theta_end,covfunc,cross_covfunc,Xtrain',Ytest,tau1,tau2,tau3,missingStack);
elseif algorithm == 5
    [fest,vest] = predictSECosFITC(theta_end,covfunc,Xtrain(~missing)',Ytest(~missing),Xtrain',numInducingFITC);
    theta_end = theta_end.cov;
elseif algorithm == 6
    [fest,vest] = predictFullGP(theta_end,covfunc,Xtrain(~missing)',Ytest(~missing),Xtrain');
    theta_end = theta_end.cov;
elseif algorithm == 7
    [fest,vest] = predictVSGP(theta_end,Xtrain(~missing)',Ytest(~missing),Xtrain',numInducingFITC);
    theta_end = theta_end.cov;
else
    error('no algorithm');
end
results(it).testTime = toc;
fprintf('Test time : %fs\n', results(it).testTime);

%% data reconstruction loss
ytrue = yOri(missing);
ynoisy = y(missing);
yreco = fest(missing);
vreco = vest(missing);

fprintf('training error : %f\n', norm(yOri(~missing) - fest(~missing), 2));
smse = smsError(ytrue,yreco);
msll = mslLoss(ynoisy,yreco,vreco+exp(theta_end(end)),mean(Ytrain),var(Ytrain));
fprintf('smse %.3f\n', smse);
fprintf('msll %.3f\n', msll);
results(it).smse = smse;
results(it).msll = msll;

%% plot
x = (1:length(Ytrain))';
figure;
hold on;

fillColor = [0.68, 0.92, 1]
ar = fill([x', fliplr(x')], [fest' - 2 * sqrt(vest'), fliplr(fest' + 2 * sqrt(vest'))], fillColor);
set(ar, 'FaceAlpha', 0.5)
set(ar, 'EdgeAlpha', 0)
%get(child, '')
plot(x, y, '-g')
plot(x, Ytrain, '-r')
plot(x, fest, '-b')

hLeg = legend('ground truth', 'train with missing bits', 'prediction');

%% plot inducing points
% inducingPointColors = {'red', 'green', 'blue', 'cyan', 'black', 'magenta', 'yellow'};
% if algorithm == 4
%     cur_tau1 = tau1;
%     for h=1:H-1
%         % h=1 : leaf
%         scaledXfint = [Xfint{h}{:}]';
%         scaledXfint = (scaledXfint - min(min(scaledXfint))) ./ (max(max(scaledXfint)) - min(min(scaledXfint)));
% 
%         for i=1:length(Xfint{h})
%             if i == length(Xfint{h})
%                 from = cur_tau1 * (i-1) + 1;
%                 to = T;
%             else
%                 from = cur_tau1 * (i-1) + 1;
%                 to = cur_tau1 * i;
%             end
%             u = splitSpace(from, to, tau2);
%             plot(u, scaledXfint(i, :) + h*0, 'LineStyle', '-', 'Marker', 'o', 'MarkerSize', 3, ...
%                 'Color', inducingPointColors{h}, 'MarkerFaceColor', inducingPointColors{h}, 'MarkerEdgeColor', 'none');
%         end
%         cur_tau1 = cur_tau1 * tau3;
%     end
% end
%% plotting inducing points end

hold off;
xlim([1000 2100])
ylim([-0.1 3])
%xlabel('t');
%ylabel('y');

% if algorithm == 1
%     title('TreeGP');
% elseif algorithm == 2
%     title('LocalGP');
% elseif algorithm == 3
%     title('HierGP (single kernel)');
% elseif algorithm == 4
%     title('HierGP (multi kernel)');
% elseif algorithm == 5
%     title('FITC');
% elseif algorithm == 6
%     title('Full GP');
% end

grid on;
set(gca,'xtick',[])
set(gca,'xticklabel',[])
set(gca,'ytick',[])
set(gca,'yticklabel',[])
set(hLeg, 'visible', 'off')
set(gca,'Position',[.05 .05 .9 .9])
%set(gca, 'xtick', [0:tau1*10:T]);
end

save(['./results/0509/' 'result_subbandHIERMULTI_' num2str(missing_length) '.mat'],'results');

% end

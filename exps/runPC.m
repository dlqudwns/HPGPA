% clear all;
% close all;
% try

rng(1);
%% Parameters
% 1: treeGP
% 2: localGP
% 3: hierGP with single cov func
% 4: hierGP with multi cov func
% 5: FITC
% 6: FullGP
% 7: VSGP
% 8: sdeKalman
pId = 1;
algorithm = 5;
% showPlot = false;

expDate = '0516';
algorithmStrs = {'tree', 'local', 'hierSingle', 'hierMulti', 'fitc', 'fullGP', 'VSGP', 'sdeKalman'};
algorithmStr = algorithmStrs{algorithm};

iter = 1;
missing_length = 64;
noEvals = 50;

if algorithm == 1 % treeGP
    params = [
        20 45 5
        20 45 11
        20 45 15
        20 45 21
        20 25 5
        20 25 11
        20 25 15
        20 15 4
        20 15 5
        20 15 7
    ];
    leaf_lengthscale = params(pId, 1);
    tau1 = params(pId, 2);
    tau2 = params(pId, 3);
    tau3 = NaN;
    numInducingFITC = NaN;
elseif algorithm == 2 %localGP
    params = [
        20 45 5
        20 45 11
        20 45 15
        20 45 21
        20 25 5
        20 25 11
        20 25 15
        20 15 4
        20 15 5
        20 15 7
    ];
    leaf_lengthscale = params(pId, 1);
    tau1 = params(pId, 2);
    tau2 = params(pId, 3);
    tau3 = NaN;
    numInducingFITC = NaN;
elseif algorithm == 4 %hierGP with multi
    params = [ % tau1 tau2 tau3 leaf_lengthscale
        20 45 5 2
        20 45 11 2
        20 45 15 2
        20 45 21 2
        20 25 5 2
        20 25 11 2
        20 25 15 2
        20 15 4 2
        20 15 5 2
        20 15 7 2
    ];
    leaf_lengthscale = params(pId, 1);
    tau1 = params(pId, 2); % the number of inputs poins per each block
    tau2 = params(pId, 3); % the number of inducing points per each block
    tau3 = params(pId, 4); % only used for hierarchical GP. block grouping factor
    numInducingFITC = NaN;
elseif algorithm == 5 %fitc
    params = [
        32 1000
        64 500
        128  300
        256  200
        512  150
        1024  120
        1500  100
        2000  70
        2500  50
        3000  40
    ];
    numInducingFITC = params(pId);
    leaf_lengthscale = params(pId, 2);
    tau1 = NaN;
    tau2 = NaN;
    tau3 = NaN;
elseif algorithm == 7 %VSGP
    params = [
        32 1000
        64 500
        128  300
        256  200
        512  150
        1024  120
        1500  100
        2000  70
        2500  50
        3000  40
    ];
    numInducingFITC = params(pId);
    leaf_lengthscale = params(pId, 2);
    tau1 = NaN;
    tau2 = NaN;
    tau3 = NaN;
elseif algorithm == 8 %VSGP
    params = [
        1
        2
        3
        4
        5
        6
        7
        8
        9
        10
    ];
    numInducingFITC = params(pId);
    leaf_lengthscale = NaN;
    tau1 = NaN;
    tau2 = NaN;
    tau3 = NaN;
else
    error('no such algorithm');
end

dataset = 'power';

results = struct(...
    'pId', pId,...
    'tau1', tau1,...
    'tau2', tau2,...
    'tau3', tau3,...
    'numInducingFITC', numInducingFITC,...
    'leaf_lengthscale', leaf_lengthscale,...
    'dataset', dataset,...
    'algorithm', algorithmStr,...
    'trainTime', NaN,...
    'testTime', NaN,...
    'smse', NaN,...
    'msll', NaN,...
    'missing_length', missing_length...
);
disp(results);

%% load data, setting
[y, yOri] = loadData(dataset);
T = length(y);

%% setting missing blocks
mInd = randi(T- 1000, 1, 50) + 500;
disp(mInd);
%[29470 60289 138166 135961 120385 158630 71485 85998 14206 122955 38344 90902 135794 28616 147636 65403 62366 ...
%91006 124297 25854 36943 50240 30536 132103 53227 102922 44781 47465 137882 32600 54935 53889 38465 82837 ...
%168275 19320 142382 87131 98715 95330 96123 27297 154126  6377 70017 114051 55534 105429 175352 142297]

missingInd = zeros(T,1);
for m = 1:length(mInd);
    j = (mInd(m))+(1:randi(1)*missing_length);
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
Xtrain = 1:min(tau1*K, T);
Ytrain = YtrainOri(Xtrain);
Ytest = YtestOri(Xtrain);
missing = missingInd(Xtrain);
if algorithm >= 1 && algorithm <= 4
    missingStack = reshape(missing,[tau1,K])';
end

covfunc = {@covSEiso};
cross_covfunc = {@crosscovSEiso};

H = computeH(K, tau3) + 1;
ind_params = eval(feval(covfunc{:})); % the number of independent free parameters. 2.

%% Set initial theta lengthscale, signal variance, noise variance
theta_init = log([leaf_lengthscale, sqrt(var(Ytrain))/2, sqrt(var(Ytrain))/2]');

%% train
tic;
if algorithm == 1
    [theta_end, ~] = trainSE(theta_init,covfunc,Xtrain',Ytrain,tau1,tau2,missingStack,noEvals);
%     theta_end = theta_init;
elseif algorithm == 2
    [theta_end, ~] = trainSELocal(theta_init,covfunc,Xtrain',Ytrain,tau1,tau2,missingStack,noEvals);
elseif algorithm == 3
    [theta_end, ~] = trainSEHier_single(theta_init,covfunc,Xtrain',Ytrain,tau1,tau2,tau3,missingStack,noEvals);
elseif algorithm == 4
    [theta_end, ~] = trainSEHier(theta_init,covfunc,cross_covfunc, Xtrain',Ytrain,tau1,tau2,tau3,missingStack,noEvals,H);
    theta_end = expand_theta(theta_end, H);
% 
%     theta_end = expand_theta(theta_tmp, H);
%     
%     theta_end = theta_tmp;
%     theta_end(5:2:end-1) = theta_end(5:2:end-1) + 1;
elseif algorithm == 5
    [theta_end, ~] = trainSEFITC(theta_init,covfunc,Xtrain(~missing)',Ytrain(~missing),numInducingFITC,noEvals);
elseif algorithm == 6
    [theta_end, ~] = trainFullGP(theta_init,covfunc,Xtrain(~missing)',Ytrain(~missing),noEvals);
elseif algorithm == 7
    [theta_end, ~] = trainVSGP(theta_init,Xtrain(~missing)',Ytrain(~missing),numInducingFITC,noEvals);
elseif algorithm == 8
    [trainingTime, testTime, theta_end, nlml, fest, vest] = evalSSMSE(Xtrain(~missing)',Ytrain(~missing)',Xtrain(missing)',numInducingFITC,theta_init,noEvals);
else
    error('no algorithm');
end
results.trainTime = toc;
if algorithm == 8, results.trainTime = trainingTime; end
results.theta_init = theta_init;
results.theta_end = theta_end;
fprintf('Training time : %fs\n', results.trainTime);

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
    [fest,vest] = predictSEFITC(theta_end,covfunc,Xtrain(~missing)',Ytest(~missing),Xtrain',numInducingFITC);
    theta_end = theta_end.cov;
elseif algorithm == 6
    [fest,vest] = predictFullGP(theta_end,covfunc,Xtrain(~missing)',Ytest(~missing),Xtrain');
    theta_end = theta_end.cov;
elseif algorithm == 7
    [fest,vest] = predictVSGP(theta_end,Xtrain(~missing)',Ytest(~missing),Xtrain',numInducingFITC);
    theta_end = theta_end.cov;
elseif algorithm ~= 8
    error('no algorithm');
end
results.testTime = toc;
if algorithm == 8, results.testTime = testTime; end
fprintf('Test time : %fs\n', results.testTime);

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
results.smse = smse;
results.msll = msll;

%% plot
if showPlot
    plotLength = min(100000, T);
    x = (1:plotLength)';
    figure;
    hold on;
    plot(x(1:plotLength), y(1:plotLength), '-g', 'lineWidth', 2)
    plot(x(1:plotLength), Ytrain(1:plotLength), '-r', 'lineWidth', 2)
    plot(x(1:plotLength), fest(1:plotLength), '-b', 'lineWidth', 2)

    legend('', 'train with missing bits', 'prediction')
    plot(x(1:plotLength), fest(1:plotLength) + 2 * sqrt(vest(1:plotLength)), '--b')
    plot(x(1:plotLength), fest(1:plotLength) - 2 * sqrt(vest(1:plotLength)), '--b')

    %% plot inducing points
%     % inducingPointColors = {'red', 'green', 'blue', 'cyan', 'black', 'magenta', 'yellow', 'red', 'green', 'blue', 'cyan', 'black', 'magenta'};
%     N = H-1;
%     inducingPointColors = {};
%     colorsList = [linspace(1, 0, N)' linspace(0, 1, N)' zeros(N, 1)];
%     colorsList = [colorsList ; zeros(N, 1) linspace(1, 1, N)' linspace(0, 1, N)'];
%     colorsList = colorsList(1:2:end, :);
%     for i=1:N
%         inducingPointColors{i} = colorsList(i, :);
%     end
%     
%     if algorithm == 4
%         cur_tau1 = tau1;
%         for h=1:H-1
%             % h=1 : leaf
%             scaledXfint = [Xfint{h}{:}]';
%             scaledXfint = (scaledXfint - min(min(scaledXfint))) ./ (max(max(scaledXfint)) - min(min(scaledXfint))) * 5;
%     
%             for i=1:length(Xfint{h})
%                 if i == length(Xfint{h})
%                     from = cur_tau1 * (i-1) + 1;
%                     to = T;
%                 else
%                     from = cur_tau1 * (i-1) + 1;
%                     to = cur_tau1 * i;
%                 end
%                 u = splitSpace(from, to, tau2);
%                 plot(u, scaledXfint(i, :) + h*0 - 5, 'LineStyle', '--', 'Marker', 'o', 'MarkerSize', 5, ...
%                     'Color', inducingPointColors{h}, 'MarkerFaceColor', inducingPointColors{h}, 'MarkerEdgeColor', 'none');
%             end
%             cur_tau1 = cur_tau1 * tau3;
%         end
%     end
    %% plotting inducing points end

    hold off;
    xlim([1 plotLength])
    % ylim([-300 2800])
    xlabel('t');
    ylabel('y');

    if algorithm == 1
        title('TreeGP');
    elseif algorithm == 2
        title('LocalGP');
    elseif algorithm == 3
        title('HierGP (single cov)');
    elseif algorithm == 4
        title('HierGP (multi cov)');
    elseif algorithm == 5
        title('FITC');
    elseif algorithm == 6
        title('Full GP');
    elseif algorithm == 7
        title('VSGP');
    else
        error('no algorithm');
    end

    grid on;
    set(gca, 'xtick', [0:tau1:T]);
end

% catch
%     disp('error occured!\n');
% end

%% Save
if ~exist(['./results/' expDate], 'dir')
    mkdir(['./results/' expDate])
end

saveFilepath = ['./results/' expDate '/result_' num2str(pId) '_' dataset '_' algorithmStr '_' num2str(missing_length) '.mat'];
save(saveFilepath, 'results');
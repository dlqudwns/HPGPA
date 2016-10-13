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
% pId = 1;
% algorithm = 5;
% showPlot = true;

expDate = '0520';
algorithmStrs = {'tree', 'local', 'hierSingle', 'hierMulti', 'fitc', 'fullGP', 'VSGP'};
algorithmStr = algorithmStrs{algorithm};

iter = 1;
missing_length = 15;
noEvals = 50;

if algorithm == 1 % treeGP
    params = [
        20 9 2
        20 9 3
        20 9 4
        20 9 5
        1 9 6
        1 9 7
        1 9 8
        20 3 1
        20 3 2
        20 3 3
    ];
    leaf_lengthscale = params(pId, 1);
    tau1 = params(pId, 2);
    tau2 = params(pId, 3);
    tau3 = NaN;
    numInducingFITC = NaN;
elseif algorithm == 2 %localGP
    params = [
        20 9 2
        20 9 3
        20 9 4
        20 9 5
        20 9 6
        20 9 7
        20 9 8
        20 3 1
        20 3 2
        20 3 3
    ];
    leaf_lengthscale = params(pId, 1);
    tau1 = params(pId, 2);
    tau2 = params(pId, 3);
    tau3 = NaN;
    numInducingFITC = NaN;
elseif algorithm == 4 %hierGP with multi
    params = [ % leaf_lengthscale tau1 tau2 tau3
        20 9 2 3 %1
        20 9 3 3
        20 9 4 3
        20 9 5 3
        20 9 6 3 %5
        20 9 7 3
        20 9 8 3
        20 3 1 3
        20 3 2 3
        20 3 3 3 %10
        20 9 2 9
        20 9 3 9
        20 9 5 9
        20 9 6 9
        20 9 7 9 %15
    ];
    leaf_lengthscale = params(pId, 1);
    tau1 = params(pId, 2); % the number of inputs poins per each block
    tau2 = params(pId, 3); % the number of inducing points per each block
    tau3 = params(pId, 4); % only used for hierarchical GP. block grouping factor
    numInducingFITC = NaN;
elseif algorithm == 5 %fitc
    params = [
        3  2500 %1
        5  2000
        7  1500
        10 1000
        13 500 %5
        16 500
        20 300
        32 300
        25 300
        28 300 %10
    ];
    numInducingFITC = params(pId);
    leaf_lengthscale = params(pId, 2);
    tau1 = NaN;
    tau2 = NaN;
    tau3 = NaN;
elseif algorithm == 7 %fitc
    params = [
        3  2500 %1
        5  2000
        7  1500
        10 1000
        13 500 %5
        16 500
        20 300
        32 300
        25 300
        28 300 %10
    ];
    numInducingFITC = params(pId);
    leaf_lengthscale = params(pId, 2);
    tau1 = NaN;
    tau2 = NaN;
    tau3 = NaN;
else
    error('no such algorithm');
end

dataset = 'terrain';

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
[Y, YOri] = loadData(dataset);
T = length(Y);

%% setting missing blocks
% mInd = [ randi(T - 3 * missing_length, 300, 2)] + missing_length; % (len x 2) matrix
% mInd = splitSpace([30 30], [930 930], 16)';
% mInd = zeros(0, 2);
% for i=22:30:T-50
%     mInd = [mInd ; 
%         [ones(length(22:30:T-50), 1)*i (22:30:T-50)']
%         ];
% end
% mIndList = [407783 493452 390370 285850 230857 121413 372251 372884 473787 310596 699430 409854 248235 803478 349317 658153 355563 449220 364644 683271 363015 242315 872258 421477 521876 473093 439886 594971 279043 303615 568123 429684 341478 514064 322165 406436 385396 758854 442466 837767 428865 876225 360235 323970 342252 446093 79305 323156 573987 876872 522490 400990 565467 503139 292440 280780 598868 532661 290278 168833 245445 398272 398222 774306 488525 323826 436656 156287 849181 249160 798376 413997 305974 122239 499493 621023 659309 563356 462148 682686 340563 603346 393790 544282 735922 310718 534430 448663 511277 351036 483819 613080 336884 167497 164855 431431 347717 218515 779221 450662 478958 430173 191770 389111 295547 282553 16688 348457 459187 87155 719719 421946 342526 556087 609183 448563 47134 490215 162080 392743 458542 521061 758037 277998 119413 821269 226308 639198 607082 124110 441577 559336 540382 727614 418030 525615 658091 662091 218483 578464 407009 496036 515697 257224 202407 422037 852016 379703 311522 335308 809079 32799 331038 222092 374979 687949 721007 514523 252123 670875 698546 458908 584343 644595 601256 526226 355746 223209 889597 323936 322902 251236 395800 121446 392147 74054 536111 639493 168386 502299 213781 627709 485608 857704 580011 884633 9702 262842 527677 595621 61783 708037 837735 459437 810580 811571 703765 528789 557195 237322 127550 282645 265235 167579 888523 200620 253799 776074 71589 553163 67643 286993 278905 430857 807888 710207 830150 483904 554495 491856 703521 306141 321020 658464 207725 550393 733124 255597 269671 596194 475358 570700 197665 594324 434083 427486 652775 635253 483301 717714 580345 353611 761013 118853 288221 123480 692115 465331 897081 744494 676212 34124 530894 490819 201405 396623 605025 561358 897956 39656 758324 713714 371465 507258 125679 673793 567228 386461 597819 551321 150489 617918 625703 546766 137700 396552 463450 371255 165989 821205 134117 474715 506627 318677 150114 443025 769009 232144 594184 650517 675309 84880 742457 641649 796425 257258 90549 30161 799557 886373 ];
% mIndList = [226873 112672 171681 191283 140336 233909 73608 222942 220830 121475 252451 187788 282800 242146 297986 213685 427777 186031 249798 163777 294310 150592 372409 420386 381216 296217 234316 214156 211743 280353 222123 195268 267838 146020 116275 189332 178076 106735 24436 464326 237186 132564 133963 240395 333086 460428 160949 255828 204518 175127 506278 108480 326565 341471 298869 493983 292641 1813 159254 56486 377230 248045 216527 46635 375150 430552 166131 316580 93063 265885 286586 422063 462653 343881 103331 270301 118356 109443 332834 276778 433670 493929 407561 155892 467163 137882 511576 35593 319666 401250 167955 354861 185256 222560 318337 263319 166748 328617 326923 377602 259128 193218 6305 68183 329661 196056 366964 352804 13777 290326 441063 174633 141684 360991 177381 405118 351853 382497 305125 403041 121524 321605 128230 446672 291752 29404 205942 68398 293072 450356 136486 186216 358604 243902 433420 90317 507698 400654 217712 166177 109308 104443 33297 491098 282579 208741 371162 14049 145521 262078 89175 137351 66598 333202 294795 247724 237297 190100 191485 9641 478785 50939 343725 90664 25129 18523 268981 486165 353752 261045 227423 321296 130092 397675 408175 236534 515166 49880 216570 494515 216943 144132 287597 368677 506537 274709 306628 347608 457480 399629 428268 459911 733 290927 516583 483123 256526 289320 314799 259526 63865 428957 509050 269265 6795 192319 344913 18934 380043 402270 384968 390028 58888 417719 393486 516776 367099 198455 466172 98129 92958 322481 48061 314280 161233 13017 511730 158217 109240 465838 461721 132810 317118 380731 465411 367062 299010 97086 501402 83617 351393 403470 433291 404133 242475 43287 113222 402809 329239 308306 349523 7732 427360 31916 209345 356532 160471 62916 516546 132742 487658 378210 348691 318143 190270 26038 442006 324307 403342 393764 36663 1512 326094 92585 397772 430161 994 503905 103683 486459 106037 434613 9335 27710 100649 492913 351822 137746 39423 516515 61044 74680 516183 37354 1116 459426 458604 81103 164083 129765 ];
% rng('shuffle');
% mIndList2 = mIndList(randperm(length(mIndList)));
mIndList = [493929 461721 293072 382497 236534 466172 35593 351393 137351 189332 108480 24436 132564 265885 37354 326923 121475 9641 174633 43287 196056 276778 371162 150592 132742 252451 242475 433291 329661 433670 159254 155892 166177 242146 318143 351853 166131 372409 1116 420386 6305 261045 486165 141684 428268 90664 511576 205942 13777 314280 381216 50939 404133 27710 344913 380043 464326 353752 318337 380731 81103 63865 516183 243902 397675 282579 430552 249798 297986 349523 321605 259526 326094 137882 118356 187788 195268 247724 240395 216527 493983 233909 217712 306628 109443 216570 144132 186031 58888 433420 341471 26038 18934 486459 430161 317118 408175 18523 501402 129765 106037 332834 462653 7732 14049 6795 62916 68183 46635 185256 343881 160949 280353 158217 213685 262078 299010 164083 270301 74680 403041 399629 428957 401250 112672 190100 211743 354861 1512 360991 427360 226873 103331 494515 116275 104443 298869 186216 133963 324307 460428 305125 314799 171681 208741 190270 358604 140336 121524 403470 509050 113222 427777 167955 366964 259128 287597 33297 294795 457480 393486 328617 516515 103683 267838 296217 733 136486 161233 397772 49880 25129 255828 73608 234316 294310 29404 441063 286586 268981 316580 459911 128230 109308 516546 100649 222942 467163 503905 450356 175127 269265 191485 163777 282800 209345 402809 465838 458604 291752 459426 378210 216943 487658 405118 321296 177381 422063 290927 352804 506537 356532 83617 193218 393764 511730 89175 326565 292641 248045 492913 220830 375150 333086 289320 407561 160471 390028 97086 465411 434613 98129 367099 256526 322481 178076 263319 166748 402270 319666 132810 137746 417719 68398 516776 227423 446672 400654 93063 333202 66598 442006 198455 377230 9335 343725 145521 506278 377602 92585 109240 204518 106735 274709 222123 491098 1813 237186 348691 237297 36663 994 515166 48061 39423 367062 290326 92958 31916 222560 308306 516583 368677 61044 56486 214156 347608 483123 90317 403342 384968 507698 130092 478785 146020 191283 351822 329239 13017 192319];
mIndList = mIndList(1:80);
mInd = zeros(0, 2);
for i=1:length(mIndList)
     [mIndx, mIndy] = compute2dindex(T, mIndList(i));
     mIndx = mIndx - floor(missing_length / 2);
     mIndy = mIndy - floor(missing_length / 2);
     if mIndx + missing_length < T && mIndy + missing_length < T && mIndx > 0 && mIndy > 0
        mInd(end+1, :) = [mIndx mIndy];
     end
end

results.mInd = mInd;
% disp(mInd');

missingInd = zeros(T); % T x T matrix
for m = 1:size(mInd, 1);
    for i=0:missing_length-1
        for j=0:missing_length-1
            missingInd(mInd(m, 1)+i, mInd(m, 2)+j) = 1;
        end
    end
end
missingInd = (missingInd == 1); % ensure to be logical matrix

%% parameters
if algorithm == 1 || algorithm == 2 || algorithm == 4
    K = floor(T/tau1);
    if tau1 * K ~= T
        error('tau1 * K ~= T');
    end
    Xtrain = [1:tau1*K ; 1:tau1*K];
    Ytrain = Y;
    Ytrain(missingInd) = 0;

    % make missing stack
    missingStack = {};
    for j=1:K
        for i=1:K
            missingStack{end+1} = missingInd((i-1)*tau1+1:i*tau1, (j-1)*tau1+1:j*tau1);
        end
    end
elseif algorithm == 5 % fitc
    Xtrain = splitSpace([1 1], [T T], T); % [1 1; 2 1; ... ; 479 480 ; 480 480]
    Xtest  = Xtrain;
    Xtrain = Xtrain(:, ~missingInd(:));
    Ytrain = Y(:);
    Ytrain = Ytrain(~missingInd(:));
    missing = missingInd(:);
end

covfunc = {@covSEiso};
cross_covfunc = {@crosscovSEiso};

if algorithm == 4 %hierGP
    H = computeH(K, tau3) + 1;
end
ind_params = eval(feval(covfunc{:})); % the number of independent free parameters. 2.

%% Set initial theta lengthscale, signal variance, noise variance
theta_init = log([leaf_lengthscale, sqrt(var(Ytrain(:)))/2, sqrt(var(Ytrain(:)))/2]');

%% train
tic;
if algorithm == 1
    [theta_end, ~] = trainSE(theta_init,covfunc,Xtrain',Ytrain,tau1,tau2,missingStack,noEvals);

%     tmp = load(sprintf('/home/jmlee/workspace/nips2016_hgpa/belief-propagation/results/0518/result_%d_terrain_tree_15.mat', pId));
%     theta_end = tmp.results.theta_end;
elseif algorithm == 2
    [theta_end, ~] = trainSELocal(theta_init,covfunc,Xtrain',Ytrain,tau1,tau2,missingStack,noEvals);

%     tmp = load(sprintf('/home/jmlee/workspace/nips2016_hgpa/belief-propagation/results/0518/result_%d_terrain_local_15.mat', pId));
%     theta_end = tmp.results.theta_end;
elseif algorithm == 4
    [theta_end, ~] = trainSEHier(theta_init,covfunc,cross_covfunc, Xtrain',Ytrain,tau1,tau2,tau3,missingStack,noEvals,H);
    theta_end = expand_theta(theta_end, H);

%     tmp = load(sprintf('/home/jmlee/workspace/nips2016_hgpa/belief-propagation/results/0518/result_%d_terrain_hierMulti_15.mat', pId));
%     theta_end = tmp.results.theta_end;
elseif algorithm == 5
    [theta_end, ~] = trainSEFITC(theta_init,covfunc,Xtrain',Ytrain,numInducingFITC,noEvals);

%     tmp = load(sprintf('/home/jmlee/workspace/nips2016_hgpa/belief-propagation/results/0518/result_%d_terrain_fitc_15.mat', pId));
%     theta_end = tmp.results.theta_end;
elseif algorithm == 7
    [theta_end, ~] = trainVSGP(theta_init,Xtrain',Ytrain,numInducingFITC,noEvals);

%     tmp = load(sprintf('/home/jmlee/workspace/nips2016_hgpa/belief-propagation/results/0518/result_%d_terrain_fitc_15.mat', pId));
%     theta_end = tmp.results.theta_end;
else
    error('no algorithm');
end
results.trainTime = toc;
results.theta_init = theta_init;
results.theta_end = theta_end;
fprintf('Training time : %fs\n', results.trainTime);

%% predict
tic;
if algorithm == 1
    [fest,vest] = predictSE(theta_end,covfunc,Xtrain',Ytrain,tau1,tau2,missingStack);
elseif algorithm == 2
    [fest,vest] = predictSELocal(theta_end,covfunc,Xtrain',Ytrain,tau1,tau2,missingStack);
elseif algorithm == 3
    [fest,vest] = predictSEHier_single(theta_end,covfunc,Xtrain',Ytest,tau1,tau2,tau3,missingStack);
elseif algorithm == 4
    [fest,vest, Xfint, ~] = predictSEHier(theta_end,covfunc,cross_covfunc,Xtrain',Ytrain,tau1,tau2,tau3,missingStack);
elseif algorithm == 5
    [fest,vest] = predictSEFITC(theta_end,covfunc,Xtrain',Ytrain,Xtest',numInducingFITC);
elseif algorithm == 7
    [fest,vest] = predictVSGP(theta_end,Xtrain',Ytrain,Xtest',numInducingFITC);
else
    error('no algorithm');
end
results.testTime = toc;
fprintf('Test time : %fs\n', results.testTime);

%% data reconstruction loss
ytrue = YOri(missingInd);
ynoisy = Y(missingInd);
yreco = fest(missingInd);
vreco = vest(missingInd);
fprintf('training error : %f\n', norm(YOri(~missingInd) - fest(~missingInd), 'fro'));

if algorithm <= 4
    smse = smsError(ytrue,yreco);
    msll = mslLoss(ynoisy,yreco,vreco+exp(theta_end(end)),mean(Ytrain(:)),var(Ytrain(:)));
elseif algorithm == 5
    smse = smsError(ytrue,yreco);
    msll = mslLoss(ynoisy,yreco,vreco+exp(2*theta_end.lik),mean(Ytrain(:)),var(Ytrain(:)));
end
fprintf('smse %.3f\n', smse);
fprintf('msll %.3f\n', msll);
results.smse = smse;
results.msll = msll;

%% plot
if showPlot
    hFig = figure;
    
    % 1.
    subplot(2,2,1);
    imagesc(Y);
    for i=1:size(mInd,1)
        rectangle('Position', [mInd(i, 2) mInd(i, 1) missing_length missing_length]);
    end
    colorbar;
    caxis([min(min(Y)) - (max(max(Y)) - min(min(Y))) * 0.1, max(max(Y)) + (max(max(Y)) - min(min(Y))) * 0.1]);
    grid on;
    set(gca, 'xtick', 0:tau1:T);
    set(gca, 'ytick', 0:tau1:T);
    
    % 2. 
    subplot(2,2,2);
    imagesc(reshape(fest, T, T));
    for i=1:size(mInd,1)
        rectangle('Position', [mInd(i, 2) mInd(i, 1) missing_length missing_length]);
    end
    colorbar;
    caxis([min(min(Y)) - (max(max(Y)) - min(min(Y))) * 0.1, max(max(Y)) + (max(max(Y)) - min(min(Y))) * 0.1]);
    grid on;
    set(gca, 'xtick', 0:tau1:T);
    set(gca, 'ytick', 0:tau1:T);
    
    % 3.
    subplot(2,2,3);
    imagesc(reshape(vest, T, T));
    for i=1:size(mInd,1)
        rectangle('Position', [mInd(i, 2) mInd(i, 1) missing_length missing_length]);
    end
    colorbar;
    grid on;
    set(gca, 'xtick', 0:tau1:T);
    set(gca, 'ytick', 0:tau1:T);
    
    % 4.
    subplot(2,2,4);
    d = Y - reshape(fest, T, T);
    imagesc(d);
    for i=1:size(mInd,1)
        rectangle('Position', [mInd(i, 2) mInd(i, 1) missing_length missing_length]);
    end
    colorbar;
    grid on;
    set(gca, 'xtick', 0:tau1:T);
    set(gca, 'ytick', 0:tau1:T);
    
    %
    set(hFig, 'Position', [500 500 1200 950]);
    title(algorithmStr);
end
%% Save
if ~exist(['./results/' expDate], 'dir')
    mkdir(['./results/' expDate])
end

saveFilepath = ['./results/' expDate '/result_' num2str(pId) '_' dataset '_' algorithmStr '_' num2str(missing_length) '.mat'];
save(saveFilepath, 'results');

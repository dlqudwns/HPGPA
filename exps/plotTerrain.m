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
algorithm = 1;

Y = loadData('terrain');
imgResult = load('results/0518/result_image.mat');
imgResult = imgResult.img;

missing_length = 15;
T = length(Y);
mInd = [147   305
   399   148
   359   229
   278   256
   361   186
   622   314
   701    94
   590   299
   665   296
   454   160
   210   340
   428   251
   670   381
   111   326
   547   402
    81   287
   576   580
   129   249
   473   336
   474   218
   516   397
   411   200
   612   504
   475   570
   671   516
   236   400
   300   315
   552   287
   326   284
   410   378
   500   298
   618   261
   288   361
   213   194
   357   153
   514   253
   193   238
   294   140
   372    27
   675   630
   254   319
   608   175
   549   177
   547   323
   655   450
   422   625
   562   214
   671   344
   391   274
   160   234
   345   688
   581   142
   695   441
   292   462
   701   403
   443   671
   305   395
   325   212
   346    71
   330   511
   178   334
     7   291
   701    57
   437   508
   435   584
   641   221
   187   428
   473   121
   522   358
    82   387
   694   572
   460   628
   515   465
   535   135
   564   364
   251   156
    86   144
   403   450
   480   373];
tau1 = 9;

% if algorithm == 1
%     fest = imgResult.tree.fest;
%     vest = imgResult.tree.vest;
%     algorithmStr = 'TreeGP';
% elseif algorithm == 2
%     fest = imgResult.local.fest;
%     vest = imgResult.local.vest;
%     algorithmStr = 'LocalGP';
% elseif algorithm == 4
%     fest = imgResult.hier.fest;
%     vest = imgResult.hier.vest;
%     algorithmStr = 'HierGP';
% elseif algorithm == 5
%     fest = imgResult.fitc.fest;
%     vest = imgResult.fitc.vest;
%     algorithmStr = 'FITC';
% else
%     error('no algorithm');
% end
%% plot
Y = Y * 100;
imgResult.fitc.fest = imgResult.fitc.fest * 100;
imgResult.tree.fest = imgResult.tree.fest * 100;
imgResult.hier.fest = imgResult.hier.fest * 100;
imgResult.local.fest = imgResult.local.fest * 100;

width = 0.165;
height = 0.85;
fontSize = 25;
colorbarwidth = 0.05;
remaining_padding = 1 - (width * 5 + colorbarwidth);
padding = remaining_padding / 6;

%% 1.
hFig = figure;
set(subplot(1,5,1), 'Position', [padding, 0.02, width, height]); % left bottom width height
imagesc(Y);
for i=1:size(mInd,1)
    rectangle('Position', [mInd(i, 2) mInd(i, 1) missing_length missing_length]);
end
% colorbar;
caxis([min(min(Y)), max(max(Y))]);
set(gca, 'xtick', 0:tau1:T);
set(gca, 'ytick', 0:tau1:T);
xlim([306 423]);
ylim([207 324]);
title('Original','FontSize',fontSize);
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);

%% 2.
set(subplot(1,5,2), 'Position', [width + 2*padding, 0.02, width height]); % left bottom width height
imagesc(reshape(imgResult.fitc.fest, T, T));
for i=1:size(mInd,1)
    rectangle('Position', [mInd(i, 2) mInd(i, 1) missing_length missing_length]);
end
% colorbar;
caxis([min(min(Y)), max(max(Y))]);
set(gca, 'xtick', 0:tau1:T);
set(gca, 'ytick', 0:tau1:T);
xlim([306 423]);
ylim([207 324]);
title('FITC (1024)','FontSize',fontSize);
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);

%% 3.
set(subplot(1,5,3), 'Position', [2*width + 3*padding, 0.02, width height]); % left bottom width height
imagesc(reshape(imgResult.local.fest, T, T));
for i=1:size(mInd,1)
    rectangle('Position', [mInd(i, 2) mInd(i, 1) missing_length missing_length]);
end
% colorbar;
caxis([min(min(Y)), max(max(Y))]);
grid on;
set(gca, 'xtick', 0:tau1:T);
set(gca, 'ytick', 0:tau1:T);
xlim([306 423]);
ylim([207 324]);
title('Local GP (81,25)','FontSize',fontSize);
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);

%% 5.
set(subplot(1,5,4), 'Position', [4*width + 5*padding, 0.02, width height]); % left bottom width height
imagesc(reshape(imgResult.hier.fest, T, T));
for i=1:size(mInd,1)
    rectangle('Position', [mInd(i, 2) mInd(i, 1) missing_length missing_length]);
end
% colorbar;
caxis([min(min(Y)), max(max(Y))]);
grid on;
set(gca, 'xtick', 0:tau1:T);
set(gca, 'ytick', 0:tau1:T);
xlim([306 423]);
ylim([207 324]);
title('HPGPA (81,25,3)','FontSize',fontSize);
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);
colorbar;


%% 4.
set(subplot(1,5,4), 'Position', [3*width + 4*padding, 0.02, width height]); % left bottom width height
imagesc(reshape(imgResult.tree.fest, T, T));
for i=1:size(mInd,1)
    rectangle('Position', [mInd(i, 2) mInd(i, 1) missing_length missing_length]);
end
% colorbar;
caxis([min(min(Y)), max(max(Y))]);
grid on;
set(gca, 'xtick', 0:tau1:T);
set(gca, 'ytick', 0:tau1:T);
xlim([306 423]);
ylim([207 324]);
title('Tree GP (81,25)','FontSize',fontSize);
set(gca,'XTickLabel',[]);
set(gca,'YTickLabel',[]);




%%
set(hFig, 'Position', [500 500 1650 320]); % x y width height
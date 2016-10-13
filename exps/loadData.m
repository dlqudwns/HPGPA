function [ y, yOri ] = loadData( dataname )
if strcmp(dataname, 'audio')
    data = load('audio_subband.mat');
    y = data.y;
    T = length(y);
    y = real(hilbert(y).*exp(-1i*2*pi*data.mu*[1:T]'));
    yOri = y;

    % only use first few samples for training and testing for now
    from = 8000; len = 50000;
    y = y(from + (1:len));
    y = resample(y,1,4);
    yOri = yOri(from + (1:len));
    yOri = resample(yOri,1,4);
    T = length(y);
elseif strcmp(dataname, 'mydata')
    T = 3200;
    y = 1:T;
    y = sqrt((T/2)^2-(y-(T/2)).^2)'/1000 + 0.5;
    yOri = y;
elseif strcmp(dataname, 'mydata2')
    T = 3200;
    y = (1:T)';
    y = sin(y / 15)*0.2 + sin(y / 300) + 2;
    yOri = y;
elseif strcmp(dataname, 'stock')
    data = dlmread('data/stock/snp500.csv', ',', 1, 0);
    % date / open / high / low / close / volume / adj close
    y = data(200:end, end);
    y = (y - min(y)) ./ (max(y) - min(y));
    y = y * 5; % 0 ~ 5
    
    y = y(1:12800);
    y = flip(y);

    yOri = y;
    T = length(y);
elseif strcmp(dataname, 'power')
    data = load('data/power/house_pc.mat');
    y = data.data(:, 1);
    y = smooth(y, 60);
    y = y(1:45*2^12); % 45 : block size...
%     y = y(1:(90 * 2^10)); %368640 = 18 * 2^14. Why 90? 90x2^4 = 1440min = 1hr
%     y = resample(y, 1, 120);
    yOri = y;
    T = length(y);
elseif strcmp(dataname, 'terrain')
    data = load('data/terrain.mat');
%     len = 960;
%     row = 200;
%     col = 200;
%     y = data.Y(row:row+len-1, col:col+len-1);
    y = data.Y(100:100+729-1, 100:100+729-1);
    y = y ./ 100;
    yOri = y;
else
    errer('no such dataset');
end

if ~strcmp(dataname, 'stock') && ~strcmp(dataname, 'terrain')
    y = y+randn(T,1)/50; % add some noise to the original data
elseif strcmp(dataname, 'terrain')
    y = y+randn(size(y))/10; % add some noise to the original data
end

end


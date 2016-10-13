function [ result ] = loadTerrainData(areanames)
data_dir_path = 'data/terrain';

%% 1. Load terrains
maps = {};

for k=1:length(areanames)
    areaname = areanames{k};

    files = dir(sprintf('%s/%s*.asc', data_dir_path, areaname));
    files = {files.name};

    if (length(files) == 0)
        error(sprintf('no files in "%s"', areaname));
    end

    for i=1:length(files)
        filename = sprintf('%s/%s', data_dir_path, files{i});
        disp(filename);
        % load infos
        fid = fopen(filename, 'r');
        fgets(fid);fgets(fid); strX = fgets(fid); strY = fgets(fid); fclose(fid);
        x = str2double(strX(11:end)) / 10000;
        y = str2double(strY(11:end)) / 10000;

        mapData = dlmread(filename, ' ', 5, 0);
        maps{end+1} = struct('x', x, 'y', y, 'data', mapData);
    end
end

%% Convert to matrix
sizeX = 200;
sizeY = 200;

minX = Inf; maxX = -Inf;
minY = Inf; maxY = -Inf;

for i=1:length(maps)
    minX = min(minX, maps{i}.x);
    minY = min(minY, maps{i}.y);
    maxX = max(maxX, maps{i}.x);
    maxY = max(maxY, maps{i}.y);
end

lenX = sizeX * (maxX - minX + 1);
lenY = sizeY * (maxY - minY + 1);

if lenX * lenY > 20000000
    error(sprintf('size of the resultant matrix is too large! : %d', lenX*lenY));
end
result = zeros(lenY, lenX);

for i=1:length(maps)
    x = maps{i}.x - minX;
    y = maxY - maps{i}.y;
    result(sizeY*y+1:sizeY*(y+1), sizeX*x+1:sizeX*(x+1)) = maps{i}.data;
end


end

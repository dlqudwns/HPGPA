% Refer to https://www.ordnancesurvey.co.uk/opendatadownload/products.html
% for list of areas (e.g. NY, NS, SP, ...)

data = loadTerrainData({'NS', 'NT'});

figure;
contour(data);

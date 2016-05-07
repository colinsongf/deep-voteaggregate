function rungmm()

% decide the number of mm's
numCenters = 5;

% load data
a = load('web/web1v.mat');
X = a.data;
X = double(X);

% fit gmm over data
options = statset('Display','final', 'maxIter', 200);
gm = fitgmdist(X,numCenters,'Options', options, 'RegularizationValue', 0.1);

% find clusters
idx = cluster(gm,X);
save('web-gmm.mat', 'idx');

end

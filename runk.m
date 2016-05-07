function [C,A,E] = runk()

numCenters = 5;
numTrials = 3;
maxNumIterations = 200;
initialization = 'plusplus';
distance = 'l2' ;

%% Create an example dataset

a = load('web/web1v.mat')
X = a.data;
X = double(X');

%% Run various k-means algorithms on the data
options = {{'Algorithm', 'Lloyd'}};

[C, A, E] = vl_kmeans(X, numCenters, 'Verbose', 'Distance', distance, 'MaxNumIterations', maxNumIterations, options{1}{:}) ;
end

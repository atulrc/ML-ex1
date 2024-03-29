%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear regression exercise.
%
%  You will need to complete the following functions in this
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear all; close all; clc

fprintf('Loading data ...\n');

%% Load Data

data = load('ex1data2.txt');
%data = load('housing.data');

[examples, features] = size(data);

X = data(:, 1:(features - 1));
y = data(:, features);

m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
%fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
[X(1:10,:), y(1:10,:)]

fprintf('Program paused. Press enter to continue.\n');
pause();

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);
fprintf('Computed mean: \n');
fprintf(' %f \n', mu);
fprintf('\n');

fprintf('Computed standard deviation: \n');
fprintf(' %f \n', sigma);
fprintf('\n');

%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha).
%
%               Your task is to first make sure that your functions -
%               computeCost and gradientDescent already work with
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Add intercept term to X
X = [ones(m, 1) X];

%fprintf(' x = [%.0f %.0f] \n', [(1:10,:)');
X(1:10,:)

% Choose some alpha value
alpha = 0.1;
num_iters = 50;

% Init Theta and Run Gradient Descent
n = size(X,2);
theta = zeros(n, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.

predict = [1, 1650, 3];
%predict = [1, 0.08014, 0.00, 5.960, 0, 0.4990, 5.8500, 41.50, 3.9342, 5, 279.0, 19.20, 396.90, 8.77];

price = (predict - [0 mu]) ./ [1 sigma] * theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent): $%f'], price);
fprintf('\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form
%               solution for linear regression using the normal
%               equations. You should complete the code in
%               normalEqn.m
%
%               After doing so, you should complete this code
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data

data = csvread('ex1data2.txt');
%data = load('housing.data');

[examples, features] = size(data);

X = data(:, 1:(features - 1));
y = data(:, features);

m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = predict * theta;


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations): $%f'], price);
fprintf('\n');

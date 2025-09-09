% Clear workspace and initialize parallel pool
clear; clc;

% Start logging to a file
diary('training_log.txt');  % Start logging to the training log file
diary on;

% Parameter Setup
numImages = 10500; % Number of images per category for training
imageTypes = {'Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag'};
totalImages = length(imageTypes) * numImages;
batchSize = 500; % Batch size for processing images

% Initialize feature and label arrays (768 RGB histograms)
features = zeros(totalImages, 768);
labels = cell(totalImages, 1);
errorLogs = cell(totalImages, 1);  % To store error messages per worker

% Start parallel pool if not already started
if isempty(gcp('nocreate'))
    parpool('Processes');
end

% Initialize progress display for Training Images
hWaitBar = waitbar(0, 'Processing training images...');

% Batch processing for loading and feature extraction (Training Images)
for batchStart = 1:batchSize:totalImages
    batchEnd = min(batchStart + batchSize - 1, totalImages);
    parfor idx = batchStart:batchEnd
        localErrorLog = '';  % Local error log for this iteration
        try
            typeIdx = ceil(idx / numImages);
            imgIdx = mod(idx - 1, numImages) + 1;

            % Construct image file path and read image
            imgPath = fullfile(imageTypes{typeIdx}, [imageTypes{typeIdx} ' (' num2str(imgIdx) ').jpg']);
            img = imread(imgPath);

            % Extract histogram features from RGB channels
            hist_red = imhist(img(:,:,1), 256);
            hist_green = imhist(img(:,:,2), 256);
            hist_blue = imhist(img(:,:,3), 256);
            img_hist = [hist_red; hist_green; hist_blue]';

            features(idx, :) = img_hist;
            labels{idx} = imageTypes{typeIdx};
        catch ME
            % Store the error message in the local variable
            localErrorLog = sprintf('Error processing image %d: %s\n', idx, ME.message);
        end

        % Save the local error log
        errorLogs{idx} = localErrorLog;

        % Log every 1000 images
        if mod(idx, 1000) == 0
            fprintf('Processed %d/%d training images...\n', idx, totalImages);
        end
    end
    waitbar(batchEnd / totalImages, hWaitBar, sprintf('Processing training images... %d/%d', batchEnd, totalImages));
end
close(hWaitBar);

% Combine error logs after the loop
finalErrorLog = '';
for i = 1:length(errorLogs)
    if ~isempty(errorLogs{i})
        finalErrorLog = strcat(finalErrorLog, errorLogs{i});
    end
end

% Write errors to log file (if any)
if ~isempty(finalErrorLog)
    logFile = fopen('error_log.txt', 'w');
    fprintf(logFile, '%s', finalErrorLog);
    fclose(logFile);
end

% Convert features to double precision and normalize them
features = double(features);
features_mean = mean(features, 1);
features_std = std(features, 0, 1);
features_std(features_std == 0) = 1; % Avoid division by zero
features_norm = (features - features_mean) ./ features_std;

% Apply PCA and retain components explaining 90% variance
features_gpu = gpuArray(features_norm);
[coeff, score, ~, ~, explained] = pca(features_gpu);
features_pca = gather(score);
num_components = find(cumsum(explained) >= 90, 1);
features_pca = features_pca(:, 1:num_components);

fprintf('PCA completed, retaining %d components.\n', num_components);

% Save features, labels, and PCA coefficients for testing phase
save('trained_model.mat', 'features_pca', 'labels', 'coeff', 'features_mean', 'features_std', 'num_components', '-v7.3');

% Scatter plot of the first two principal components (Training Data)
figure;
gscatter(features_pca(:, 1), features_pca(:, 2), labels);
title('PCA: First two principal components of training data');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
legend(imageTypes, 'Location', 'bestoutside');
grid on;

% Hyperparameter Tuning and Training for SVM Kernels with Parallel Processing
    % Updated Hyperparameter Tuning for SVM Kernels with Improved Polynomial Kernel
kernels = {'linear'};
best_params = struct();
svm_models = struct(); % Store trained SVM models for testing
cv_accuracies = zeros(length(kernels), 1);

% General Hyperparameter tuning options with increased evaluations for polynomial kernel
opts_general = struct('Optimizer', 'bayesopt', 'ShowPlots', false, 'Verbose', 1, ...
                      'MaxObjectiveEvaluations', 5, 'UseParallel', true);

for k = 1:length(kernels)
    fprintf('Tuning hyperparameters for SVM (%s kernel)...\n', kernels{k});
    
    % Define hyperparameter grid based on the kernel
    if strcmp(kernels{k}, 'linear')
        paramGrid = {'BoxConstraint'};
        opts = opts_general;  % use general options
    end
    
    % Fit the SVM model with ECOC for multi-class classification and optimize hyperparameters
    svm_model = fitcecoc(features_pca, labels, 'Learners', templateSVM('KernelFunction', kernels{k}), ...
                         'OptimizeHyperparameters', paramGrid, 'HyperparameterOptimizationOptions', opts);

    % Extract the best parameters and model
    best_params.(kernels{k}) = svm_model.HyperparameterOptimizationResults.XAtMinObjective;
    svm_models.(kernels{k}) = svm_model;  % Store the trained model
    
    fprintf('%s Kernel: Best Hyperparameters found.\n', kernels{k});
    
    % Evaluate cross-validation accuracy
    cv_model = crossval(svm_model, 'KFold', 5); % Perform cross-validation separately
    cv_accuracy = 1 - kfoldLoss(cv_model);
    cv_accuracies(k) = cv_accuracy * 100;  % Store accuracy
    fprintf('SVM (%s kernel) cross-validation accuracy: %.2f%%\n', kernels{k}, cv_accuracy * 100);
end


% Save the trained SVM models and the best hyperparameters
save('svm_models.mat', 'svm_models', 'best_params', '-v7.3');

% Plot cross-validation accuracies
figure;
bar(cv_accuracies);
set(gca, 'XTickLabel', kernels);
ylabel('Accuracy (%)');
title('Cross-Validation Accuracy for SVM Kernels');
grid on;

% Close parallel pool
delete(gcp('nocreate'));

% Stop logging
diary off;

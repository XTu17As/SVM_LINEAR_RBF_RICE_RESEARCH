% Start logging to a file
diary('testing_log.txt');  % Start logging to the testing log file
diary on;

% Load the trained model, PCA coefficients, and training stats
load('trained_model.mat', 'coeff', 'features_mean', 'features_std', 'num_components');
load('svm_models.mat', 'svm_models');

% Parameter Setup
numTestImages = 4500; % Number of test images per category
imageTypes = {'Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag'};
numClasses = length(imageTypes);
totalTestImages = numClasses * numTestImages;
batchSize = 500; % Batch size for processing images (if needed)

% Initialize feature and label arrays for test images
test_features = zeros(totalTestImages, 256 * 3); % 768 features
test_labels = cell(totalTestImages, 1);
testErrorLogs = cell(totalTestImages, 1);  % To store error messages per worker

% Initialize waitbar for feature extraction
hWaitBarFeature = waitbar(0, 'Extracting features for test images...');

% Feature Extraction using parfor
parfor idx = 1:totalTestImages
    try
        typeIdx = ceil(idx / numTestImages);
        imgIdx = mod(idx - 1, numTestImages) + 10501; % Start from 1001 for test images

        % Construct image file path
        imgPath = fullfile(imageTypes{typeIdx}, [imageTypes{typeIdx} ' (' num2str(imgIdx) ').jpg']);
        
        % Read the image
        img = imread(imgPath);

        % Extract histogram features for each color channel
        hist_red = imhist(img(:,:,1), 256);
        hist_green = imhist(img(:,:,2), 256);
        hist_blue = imhist(img(:,:,3), 256);
        
        % Concatenate histograms into a single feature vector
        img_hist = [hist_red; hist_green; hist_blue]';

        % Store features and labels
        test_features(idx, :) = img_hist;
        test_labels{idx} = imageTypes{typeIdx};
    catch ME
        % Store the error message in the local variable
        testErrorLogs{idx} = sprintf('Error processing test image %d: %s\n', idx, ME.message);
    end
end

% Update waitbar to indicate feature extraction completion
close(hWaitBarFeature);
disp('Feature extraction completed.');

% Combine error logs after the loop
finalTestErrorLog = '';
for i = 1:length(testErrorLogs)
    if ~isempty(testErrorLogs{i})
        finalTestErrorLog = strcat(finalTestErrorLog, testErrorLogs{i});
    end
end

% Write errors to log file (if any)
if ~isempty(finalTestErrorLog)
    logFileTest = fopen('error_log_test.txt', 'w');
    fprintf(logFileTest, '%s', finalTestErrorLog);
    fclose(logFileTest);
    fprintf('Errors encountered during feature extraction. Check error_log_test.txt for details.\n');
else
    fprintf('No errors encountered during feature extraction.\n');
end

% Normalize and apply PCA to test features
test_features = double(test_features);
test_features_norm = (test_features - features_mean) ./ features_std;
test_features_pca = test_features_norm * coeff(:, 1:num_components);

% Evaluate the trained SVM models on test data
kernels = {'linear'};
for k = 1:length(kernels)
    fprintf('=====================================\n');
    fprintf('          Testing SVM (%s kernel)\n', kernels{k});
    fprintf('=====================================\n');

    % Predict labels using the trained SVM model
    predicted_labels = predict(svm_models.(kernels{k}), test_features_pca);

    % Compute confusion matrix
    confMat = confusionmat(test_labels, predicted_labels);

    % Display confusion matrix table with explanation
    fprintf('Confusion Matrix (Actual vs Predicted):\n');
    fprintf('-------------------------------------------------\n');
    fprintf('Actual \\ Predicted\t');
    for j = 1:numClasses
        fprintf('%-10s\t', imageTypes{j});
    end
    fprintf('\n');

    for i = 1:numClasses
        fprintf('%-10s\t', imageTypes{i});
        for j = 1:numClasses
            fprintf('%d\t\t', confMat(i, j));
        end
        fprintf('\n');
    end
    fprintf('-------------------------------------------------\n');
    fprintf('Where:\n');
    fprintf('TP = True Positives: Correct predictions for each class.\n');
    fprintf('FP = False Positives: Instances incorrectly predicted as a class.\n');
    fprintf('FN = False Negatives: Instances incorrectly classified as another class.\n\n');

    % Initialize variables to store metrics
    precision = zeros(numClasses, 1);
    recall = zeros(numClasses, 1);
    f1_score = zeros(numClasses, 1);
    accuracy = zeros(numClasses, 1);

    % Calculate precision, recall, F1-score, and accuracy for each class
    for i = 1:numClasses
        TP = confMat(i, i);  % True Positives
        FP = sum(confMat(:, i)) - TP;  % False Positives
        FN = sum(confMat(i, :)) - TP;  % False Negatives
        TN = sum(confMat(:)) - (TP + FP + FN);  % True Negatives

        % Handle division by zero
        precision(i) = TP / (TP + FP + eps);
        recall(i) = TP / (TP + FN + eps);
        f1_score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
        accuracy(i) = (TP + TN) / (TP + TN + FP + FN) * 100;

        % Print results for each class
        fprintf('%-12s\t %.2f%%\n', imageTypes{i}, accuracy(i));
        fprintf('   F1: %.2f, Recall: %.2f, Precision: %.2f\n\n', f1_score(i), recall(i), precision(i));
    end

    % Overall accuracy
    overall_accuracy = sum(diag(confMat)) / sum(confMat(:)) * 100;
    fprintf('Overall Accuracy\t %.2f%%\n', overall_accuracy);
    fprintf('   F1: %.2f, Recall: %.2f, Precision: %.2f\n', mean(f1_score), mean(recall), mean(precision));
    
    % Display confusion matrix as a heatmap
    figure;
    confusionchart(confMat, imageTypes);
    title(sprintf('Confusion Matrix (%s Kernel)', kernels{k}));
    saveas(gcf, sprintf('confusion_matrix_%s_kernel.png', kernels{k}));
    close(gcf); % Close the figure to prevent accumulation

    % Log the per-class metrics into a log file
    logFileTestResults = fopen('test_results_log.txt', 'a');
    fprintf(logFileTestResults, '=====================================\n');
    fprintf(logFileTestResults, '          Testing SVM (%s kernel)\n', kernels{k});
    fprintf(logFileTestResults, '=====================================\n');
    for i = 1:numClasses
        fprintf(logFileTestResults, '%-12s\t %.2f%%\n', imageTypes{i}, accuracy(i));
        fprintf(logFileTestResults, '   F1: %.2f, Recall: %.2f, Precision: %.2f\n\n', f1_score(i), recall(i), precision(i));
    end
    fprintf(logFileTestResults, 'Overall Accuracy\t %.2f%%\n', overall_accuracy);
    fprintf(logFileTestResults, '   F1: %.2f, Recall: %.2f, Precision: %.2f\n', mean(f1_score), mean(recall), mean(precision));
    fclose(logFileTestResults);

    % PCA Scatter Plot
    figure;
    gscatter(test_features_pca(:, 1), test_features_pca(:, 2), predicted_labels);
    title(sprintf('PCA: First Two Principal Components (Test Data - %s Kernel)', kernels{k}));
    xlabel('Principal Component 1');
    ylabel('Principal Component 2');
    legend(imageTypes, 'Location', 'bestoutside');
    grid on;

    % Save PCA plot as PNG
    saveas(gcf, sprintf('pca_test_data_%s_kernel.png', kernels{k}));
    close(gcf); % Close the figure to prevent accumulation

    % Plot Precision, Recall, F1 Score
    figure;
    bar([precision, recall, f1_score], 'grouped');
    set(gca, 'XTickLabel', imageTypes);
    title(sprintf('SVM (%s Kernel) - Performance Metrics', kernels{k}));
    ylabel('Score');
    legend({'Precision', 'Recall', 'F1-Score'}, 'Location', 'bestoutside');
    grid on;

    % Save the bar plot as PNG
    saveas(gcf, sprintf('performance_metrics_%s_kernel.png', kernels{k}));
    close(gcf); % Close the figure to prevent accumulation
end

disp('Testing complete. Results saved to test_results_log.txt and error_log_test.txt');
diary off;  % Stop logging

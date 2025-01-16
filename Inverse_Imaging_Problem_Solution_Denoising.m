% Step 1: Load and Preprocess Image Data using File Dialog
[file, path] = uigetfile({'*.jpg;*.png;*.jpeg;*.bmp','Image Files (*.jpg, *.png, *.jpeg, *.bmp)'}, ...
                         'Select an Image');
if isequal(file, 0)
    disp('User selected Cancel');
    return;
else
    inputImage = imread(fullfile(path, file)); % Load the selected image
    disp(['User selected ', fullfile(path, file)]);
end

% Preprocess the image
inputImage = im2double(inputImage); % Convert image to double precision
if size(inputImage, 3) == 1
    inputImage = repmat(inputImage, [1 1 3]); % Convert grayscale to RGB if needed
end

% Step 2: Add noise to simulate forward operator
noisyImage = imnoise(inputImage, 'gaussian', 0, 0.02); % Add Gaussian noise

% Display input and noisy images
figure;
subplot(1,2,1); imshow(inputImage); title('Original Image');
subplot(1,2,2); imshow(noisyImage); title('Noisy Image');

% Step 3: U-Net Architecture Definition
inputSize = [256 256 3];
numClasses = 3; % For RGB output

% Encoder (Downsampling Path)
layers = [
    imageInputLayer(inputSize, 'Name', 'input')
    
    % Downsampling blocks
    convolution2dLayer(3,64,'Padding','same','Name','conv1_1')
    reluLayer('Name','relu1_1')
    convolution2dLayer(3,64,'Padding','same','Name','conv1_2')
    reluLayer('Name','relu1_2')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool1')
    
    convolution2dLayer(3,128,'Padding','same','Name','conv2_1')
    reluLayer('Name','relu2_1')
    convolution2dLayer(3,128,'Padding','same','Name','conv2_2')
    reluLayer('Name','relu2_2')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool2')

    convolution2dLayer(3,256,'Padding','same','Name','conv3_1')
    reluLayer('Name','relu3_1')
    convolution2dLayer(3,256,'Padding','same','Name','conv3_2')
    reluLayer('Name','relu3_2')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool3')

    convolution2dLayer(3,512,'Padding','same','Name','conv4_1')
    reluLayer('Name','relu4_1')
    convolution2dLayer(3,512,'Padding','same','Name','conv4_2')
    reluLayer('Name','relu4_2')
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool4')

    % Bottleneck
    convolution2dLayer(3,1024,'Padding','same','Name','conv5_1')
    reluLayer('Name','relu5_1')
    convolution2dLayer(3,1024,'Padding','same','Name','conv5_2')
    reluLayer('Name','relu5_2')
    
    % Upsampling Path (Decoder)
    transposedConv2dLayer(2,512,'Stride',2,'Cropping','same','Name','upconv4')
    convolution2dLayer(3,512,'Padding','same','Name','conv6_1')
    reluLayer('Name','relu6_1')
    convolution2dLayer(3,512,'Padding','same','Name','conv6_2')
    reluLayer('Name','relu6_2')
    
    transposedConv2dLayer(2,256,'Stride',2,'Cropping','same','Name','upconv3')
    convolution2dLayer(3,256,'Padding','same','Name','conv7_1')
    reluLayer('Name','relu7_1')
    convolution2dLayer(3,256,'Padding','same','Name','conv7_2')
    reluLayer('Name','relu7_2')
    
    transposedConv2dLayer(2,128,'Stride',2,'Cropping','same','Name','upconv2')
    convolution2dLayer(3,128,'Padding','same','Name','conv8_1')
    reluLayer('Name','relu8_1')
    convolution2dLayer(3,128,'Padding','same','Name','conv8_2')
    reluLayer('Name','relu8_2')
    
    transposedConv2dLayer(2,64,'Stride',2,'Cropping','same','Name','upconv1')
    convolution2dLayer(3,64,'Padding','same','Name','conv9_1')
    reluLayer('Name','relu9_1')
    convolution2dLayer(3,64,'Padding','same','Name','conv9_2')
    reluLayer('Name','relu9_2')

    % Output layer
    convolution2dLayer(1, numClasses, 'Name', 'conv10')
    regressionLayer('Name','output') % For regression since it's a denoising task
];

% Step 4: Define Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',2500, ...
    'MiniBatchSize',16, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false);

% Step 5: Prepare Training Data
% Resize the images to fit the network's input size
noisyImage = imresize(noisyImage, [256 256]);
inputImage = imresize(inputImage, [256 256]);

XTrain = {noisyImage}; % Noisy images as input
YTrain = {inputImage}; % Clean images as ground truth

% Step 6: Train the U-Net Model
net = trainNetwork(cat(4, XTrain{:}), cat(4, YTrain{:}), layers, options);

% Step 7: Evaluate the Model on the Test Image
denoisedImage = predict(net, noisyImage);

% Step 8: Display Results
figure;
imshowpair(noisyImage, denoisedImage, 'montage');
title('Noisy Image (Left) vs Denoised Image (Right)');

% Step 9: Save Noisy and Denoised Images
outputPath = uigetdir('C:\', 'Select a Folder to Save the Images'); % Select directory to save images

if isequal(outputPath, 0)
    disp('User canceled saving');
else
    noisyImageFile = fullfile(outputPath, 'noisyImage.png');
    denoisedImageFile = fullfile(outputPath, 'denoisedImage.png');
    
    % Save the images
    imwrite(noisyImage, noisyImageFile);
    imwrite(denoisedImage, denoisedImageFile);
    
    disp(['Noisy image saved to: ', noisyImageFile]);
    disp(['Denoised image saved to: ', denoisedImageFile]);
end
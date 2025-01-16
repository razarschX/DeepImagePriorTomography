% MATLAB Script: Radon Transform and Image Reconstruction

% Step 1: Load a user-provided image
function main()
    % Prompt the user for the image file path
    image_path = input('Enter the path to the image file: ', 's');

    try
        original_image = imread(image_path);

        % Convert to grayscale if the image is RGB
        if size(original_image, 3) == 3
            original_image = rgb2gray(original_image);
        end

        % Normalize the image to [0, 1] for consistency
        original_image = double(original_image) / 255;

        % Warn about large images
        [rows, cols] = size(original_image);
        if rows > 256 || cols > 256
            disp('Warning: Large images may significantly slow down processing.');
        end

        % Step 2: Define theta values for Radon Transform
        theta = linspace(0, 180, max(size(original_image))); % Angles from 0 to 180 degrees

        % Step 3: Perform Radon Transform
        sinogram = radon(original_image, theta);

        % Step 4: Perform Inverse Radon Transform
        reconstructed_image = iradon(sinogram, theta, 'linear', 'Ram-Lak', 1, max(rows, cols));

        % Step 5: Visualize Results
        visualize_results(original_image, sinogram, reconstructed_image);

        % Step 6: Compare Original and Reconstructed Images
        compare_images(original_image, reconstructed_image);

    catch ME
        disp(['Error: ' ME.message]);
        disp('Exiting program...');
    end
end

% Step 2: Visualize Results
function visualize_results(original_image, sinogram, reconstructed_image)
    % Display the original image, sinogram, and reconstructed image
    figure;
    subplot(1, 3, 1);
    imshow(original_image, []);
    title('Original Image');

    subplot(1, 3, 2);
    imagesc(sinogram);
    colormap(gray);
    colorbar;
    title('Sinogram (Radon Transform)');
    xlabel('Angles (degrees)');
    ylabel('Projection position');

    subplot(1, 3, 3);
    imshow(reconstructed_image, []);
    title('Reconstructed Image (Inverse Radon)');
end

% Step 3: Compare Images
function compare_images(original_image, reconstructed_image)
    % Resize the original image to match reconstructed dimensions
    [rows_original, cols_original] = size(original_image);
    [rows_reconstructed, cols_reconstructed] = size(reconstructed_image);

    if rows_original ~= rows_reconstructed || cols_original ~= cols_reconstructed
        disp('Warning: Image dimensions do not match. Resizing reconstructed image to match the original image.');
        reconstructed_image = imresize(reconstructed_image, [rows_original, cols_original]);
    end

    % Normalize reconstructed image to [0, 1] for consistency
    reconstructed_image = (reconstructed_image - min(reconstructed_image(:))) / ...
                          (max(reconstructed_image(:)) - min(reconstructed_image(:)));

    % Compute Mean Squared Error (MSE)
    mse_value = immse(original_image, reconstructed_image);
    disp(['Mean Squared Error (MSE) between Original and Reconstructed Images: ', num2str(mse_value)]);
end
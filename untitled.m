%{
% Load an image to introduce distortion 
input_image = imread("test.jpg");

% Display the original image
figure;
imshow(input_image);
title('Original Image');

% Set parameters for barrel distortion
k_1 = 0.15; % Barrel distortion coefficient (0.15 for example)
k_2 = 0;    % Set all other distortion coefficients to zero
p_1 = 0;
p_2 = 0;

% Get image dimensions
[height, width, channels] = size(input_image);

% Initialize the distorted image with the same dimensions and type
distorted_image = uint8(zeros(size(input_image)));

% Calculate the center of the image
center_x = width / 2;
center_y = height / 2;

% Apply barrel distortion to the image
for y = 1:height
    for x = 1:width
        % Calculate the normalized coordinates relative to the center
        x_normalized = (x - center_x) / center_x;
        y_normalized = (y - center_y) / center_y;
        
        % Calculate the radial distance from the center
        r = sqrt(x_normalized^2 + y_normalized^2);
        
        % Apply the distortion equation
        x_distorted = x_normalized * (1 + k_1 * r^2 + k_2 * r^4) + 2 * p_1 * x_normalized * y_normalized + p_2 * (r^2 + 2 * x_normalized^2);
        y_distorted = y_normalized * (1 + k_1 * r^2 + k_2 * r^4) + 2 * p_2 * x_normalized * y_normalized + p_1 * (r^2 + 2 * y_normalized^2);
        
        % Convert back to pixel coordinates
        x_distorted_pixel = round(center_x * x_distorted + center_x);
        y_distorted_pixel = round(center_y * y_distorted + center_y);
        
        % Check if the distorted coordinates are within bounds
        if x_distorted_pixel >= 1 && x_distorted_pixel <= width && y_distorted_pixel >= 1 && y_distorted_pixel <= height
            for c = 1:channels
                % Copy pixel values from the input image to the distorted image
                distorted_image(y, x, c) = input_image(y_distorted_pixel, x_distorted_pixel, c);
            end
        end
    end
end

% Display the distorted image
figure;
imshow(distorted_image);
title('Distorted Image (Barrel Distortion)');

% Initialize the undistorted image with the same dimensions and type
undistorted_image = uint8(zeros(size(input_image)));

% Apply the inverse distortion to correct
for y = 1:height
    for x = 1:width
        % Calculate the normalized coordinates relative to the center
        x_normalized = (x - center_x) / center_x;
        y_normalized = (y - center_y) / center_y;
        
        % Calculate the radial distance from the center
        r = sqrt(x_normalized^2 + y_normalized^2);
        
        % Calculate the inverse distortion equation
        x_undistorted = x_normalized / (1 + k_1 * r^2 + k_2 * r^4) - 2 * p_1 * x_normalized * y_normalized / (1 + k_1 * r^2 + k_2 * r^4) - p_2 * (r^2 + 2 * x_normalized^2) / (1 + k_1 * r^2 + k_2 * r^4);
        y_undistorted = y_normalized / (1 + k_1 * r^2 + k_2 * r^4) - 2 * p_2 * x_normalized * y_normalized / (1 + k_1 * r^2 + k_2 * r^4) - p_1 * (r^2 + 2 * y_normalized^2) / (1 + k_1 * r^2 + k_2 * r^4);
        
        % Convert back to pixel coordinates
        x_undistorted_pixel = round(center_x * x_undistorted + center_x);
        y_undistorted_pixel = round(center_y * y_undistorted + center_y);
        
        % Check if the undistorted coordinates are within bounds
        if x_undistorted_pixel >= 1 && x_undistorted_pixel <= width && y_undistorted_pixel >= 1 && y_undistorted_pixel <= height
            for c = 1:channels
                % Copy pixel values from the distorted image to the undistorted image
                undistorted_image(y, x, c) = distorted_image(y_undistorted_pixel, x_undistorted_pixel, c);
            end
        end
    end
end

% Display the undistorted image
figure;
imshow(undistorted_image);
title('Undistorted Image');
%}

% Load an image to introduce distortion 
input_image = imread("test.jpg");

% Display the original image
figure;
imshow(input_image);
title('Original Image');

% Set parameters for barrel distortion
k_1 = 0.25; % Barrel distortion coefficient (0.15 for example)
k_2 = 0;    % Set all other distortion coefficients to zero
p_1 = 0;
p_2 = 0;

% Get image dimensions
[height, width, channels] = size(input_image);

% Initialize the distorted image with the same dimensions and type
distorted_image = uint8(zeros(size(input_image)));

% Calculate the center of the image
center_x = width / 2;
center_y = height / 2;

% Apply barrel distortion to the image
for y = 1:height
    for x = 1:width
        % Calculate the normalized coordinates relative to the center
        x_normalized = (x - center_x) / center_x;
        y_normalized = (y - center_y) / center_y;
        
        % Calculate the radial distance from the center
        r = sqrt(x_normalized^2 + y_normalized^2);
        
        % Apply the distortion equation with minimal overcorrection
        x_distorted = x_normalized * (1 + k_1 * r^2) + p_1 * (r^2 + 2 * x_normalized^2);
        y_distorted = y_normalized * (1 + k_1 * r^2) + p_2 * (r^2 + 2 * y_normalized^2);
        
        % Convert back to pixel coordinates
        x_distorted_pixel = round(center_x * x_distorted + center_x);
        y_distorted_pixel = round(center_y * y_distorted + center_y);
        
        % Check if the distorted coordinates are within bounds
        if x_distorted_pixel >= 1 && x_distorted_pixel <= width && y_distorted_pixel >= 1 && y_distorted_pixel <= height
            for c = 1:channels
                % Copy pixel values from the input image to the distorted image
                distorted_image(y, x, c) = input_image(y_distorted_pixel, x_distorted_pixel, c);
            end
        end
    end
end

% Display the distorted image
figure;
imshow(distorted_image);
title('Distorted Image (Barrel Distortion)');


% Initialize the undistorted image with the same dimensions and type
undistorted_image = uint8(zeros(size(input_image)));

% Apply the inverse distortion to correct
for y = 1:height
    for x = 1:width
        % Calculate the normalized coordinates relative to the center
        x_normalized = (x - center_x) / center_x;
        y_normalized = (y - center_y) / center_y;
        
        % Calculate the radial distance from the center
        r = sqrt(x_normalized^2 + y_normalized^2);
        
        % Apply the inverse distortion equation
        x_undistorted = x_normalized / (1 + k_1 * r^2) - p_1 * (r^2 + 2 * x_normalized^2);
        y_undistorted = y_normalized / (1 + k_1 * r^2) - p_2 * (r^2 + 2 * y_normalized^2);
        
        % Convert back to pixel coordinates
        x_undistorted_pixel = round(center_x * x_undistorted + center_x);
        y_undistorted_pixel = round(center_y * y_undistorted + center_y);
        
        % Check if the undistorted coordinates are within bounds
        if x_undistorted_pixel >= 1 && x_undistorted_pixel <= width && y_undistorted_pixel >= 1 && y_undistorted_pixel <= height
            for c = 1:channels
                % Copy pixel values from the distorted image to the undistorted image
                undistorted_image(y, x, c) = distorted_image(y_undistorted_pixel, x_undistorted_pixel, c);
            end
        end
    end
end

% Define a custom convolution kernel (e.g., a simple box blur)
kernel = [1, 1, 1; 1, 1, 1; 1, 1, 1] / 9; % Normalization for box blur

% Get image dimensions
[height, width, ~] = size(undistorted_image);

% Initialize the output image
output_image = uint8(zeros(height, width, 3));

% Apply convolution to each channel (R, G, B)
for c = 1:3 % Loop over color channels
    for y = 2:(height - 1) % Exclude border pixels
        for x = 2:(width - 1) % Exclude border pixels
            % Initialize the result for the current pixel
            result = 0;
            
            % Apply the kernel to the neighborhood of the pixel
            for ky = -1:1
                for kx = -1:1
                    % Calculate the coordinates of the neighboring pixel
                    neighbor_x = x + kx;
                    neighbor_y = y + ky;
                    
                    % Get the pixel value of the neighbor
                    neighbor_value = double(undistorted_image(neighbor_y, neighbor_x, c));
                    
                    % Apply the kernel value to the neighbor and accumulate
                    result = result + neighbor_value * kernel(ky + 2, kx + 2);
                end
            end
            
            % Store the result in the output image
            output_image(y, x, c) = uint8(result);
        end
    end
end
%{
% Display the output image
figure;
imshow(output_image);
title('Convolved Image');

% Define a sharpening kernel (adjust this kernel for different levels of sharpening)
sharpening_kernel = [0, -1, 0; -1, 2, -1; 0, -1, 0]; %[0, -1, 0; -1, 5, -1; 0, -1, 0];

% Apply convolution to sharpen the undistorted image
sharpened_image = imfilter(undistorted_image, sharpening_kernel);

% Display the sharpened image
figure;
imshow(sharpened_image);
title('Sharpened Undistorted Image');
%}

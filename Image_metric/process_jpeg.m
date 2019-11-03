clear all; close all; clc
JPEG_Quality = 20;
im = imread('F:\dataset\benchmark_noise\classic5\peppers.bmp');
%% work on luminance only
if size(im,3)>1
    im = rgb2ycbcr(im);
end
im_y = im(:,:,1);
im_gnd = im2double(im_y);
[hei,wid,channels] = size(im_gnd);

%% generate JPEG-compressed input
imwrite(im, 'C:\Users\Administrator\Desktop\peppers.jpg','Quality', JPEG_Quality);

%ImHR = imread('kodim01_HQ_N10.png')
%ImHR = single(ImHR); % 0-255
%sigma = 10
%ImHRNoise = ImHR + single(sigma*randn(size(ImHR))); % 0-255
%ImHRNoise = uint8(ImHRNoise); % 0-255
%imwrite(ImHRNoise, 'C:\Users\Administrator\Desktop\kodim01_noise.png')
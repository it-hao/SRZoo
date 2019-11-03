function Prepare_Train_Color_Noise_HQ_LQ()
clear all; close all; clc
path_original = 'F:/sh/data_noise/DIV2K/DIV2K_train_Color_HQ';
path_output = 'F:/sh/data_noise/DIV2K/DIV2K_train_Color_LQ';
% path_original = 'F:/dataset/benchmark_noise/Kodak24';
% path_output = 'F:/dataset/benchmark_noise/Kodak24CNoise'
ext = {'*.jpg', '*.png', '*.bmp'};
noise_all = [10, 30, 50, 70];

fprintf('Processing %s:/n');
filepaths = [];
for idx_ext = 1:length(ext)
    filepaths = cat(1, filepaths, dir(fullfile(path_original,ext{idx_ext})));
end
for idx_im = 1:length(filepaths)
    name_im = filepaths(idx_im).name;
    fprintf('%d. %s: ', idx_im, name_im);
    im_ori = imread(fullfile(path_original, name_im));
    for noise = noise_all
        fprintf('x%d ', noise);
        folder_LQ = fullfile(path_output, ['N', num2str(noise)]);
        im_LQ = add_awgn_noise(im_ori, noise);
        if ~exist(folder_LQ)
            mkdir(folder_LQ)
        end
        fn_LQ = fullfile(path_output, ['N', num2str(noise)], [name_im(1:end-4), 'n', num2str(noise), '.png']);
        imwrite(im_LQ, fn_LQ, 'png');
    end
    fprintf('\n');
end
fprintf('\n');
end

function [img] = add_awgn_noise(img, noise_level)
label = im2double(img);
% add noise
noise = bsxfun(@times,randn(size(label)),permute(noise_level/255,[3 4 1 2]));
img = single(label + noise);
end














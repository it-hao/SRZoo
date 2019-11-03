function Prepare_Test_Color_Noise_HQ_LQ()
clear all; close all; clc
path_original = '.\OriginalTestData';
dataset  = {'Kodak24', 'BSD68', 'Urban100'};
ext = {'*.jpg', '*.png', '*.bmp'};

noise_all = [10, 30, 50, 70];

for idx_set = 1:length(dataset)
    fprintf('Processing %s:/n', dataset{idx_set});
    filepaths = [];
    for idx_ext = 1:length(ext)
        filepaths = cat(1, filepaths, dir(fullfile(path_original, dataset{idx_set}, ext{idx_ext})));
    end
    for idx_im = 1:length(filepaths)
        name_im = filepaths(idx_im).name;
        fprintf('%d. %s: ', idx_im, name_im);
        im_HQ = imread(fullfile(path_original, dataset{idx_set}, name_im));
        for noise = noise_all
            fprintf('x%d ', noise);
            im_LQ = add_awgn_noise(im_HQ, noise);
            folder_HQ = fullfile('.\HQ', dataset{idx_set}, ['N', num2str(noise)]);
            folder_LQ = fullfile('.\LQ', dataset{idx_set}, ['N', num2str(noise)]);
            
            if ~exist(folder_HQ)
                mkdir(folder_HQ)
            end
            if ~exist(folder_LQ)
                mkdir(folder_LQ)
            end
            % fn
            fn_HQ = fullfile('.\HQ', dataset{idx_set}, ['N', num2str(noise)], [name_im(1:end-4), '_HQ_N', num2str(noise), '.png']);
            fn_LQ = fullfile('.\LQ', dataset{idx_set}, ['N', num2str(noise)], [name_im(1:end-4), '_LQ_N', num2str(noise), '.png']);
            imwrite(im_HQ, fn_HQ, 'png');
            imwrite(im_LQ, fn_LQ, 'png');
        end
        fprintf('/n');
    end
    fprintf('/n');
end
end

function [img] = add_awgn_noise(img, noise_level)
label = im2double(img);
% add noise
noise = bsxfun(@times,randn(size(label)),permute(noise_level/255,[3 4 1 2]));
img = single(label + noise);
end
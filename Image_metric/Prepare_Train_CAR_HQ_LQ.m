function Prepare_Train_CAR_HQ_LQ()
clear all; close all; clc
path_original = './DIV2K/DIV2K_train_Color_HQ';
path_output_LQ = './DIV2K/DIV2K_train_Gray_LQ';
path_output_HQ = './DIV2K/DIV2K_train_Gray_HQ';

ext = {'*.jpg', '*.png', '*.bmp'};
quality_all = [10, 20, 30, 40];

fprintf('Processing %s:/n');
filepaths = [];
for idx_ext = 1:length(ext)
    filepaths = cat(1, filepaths, dir(fullfile(path_original,ext{idx_ext})));
end
for idx_im = 1:length(filepaths)
    name_im = filepaths(idx_im).name;
    fprintf('%d. %s: ', idx_im, name_im);
    img = imread(fullfile(path_original, name_im));
    if size(img, 3)>1
          img = rgb2ycbcr(img);
          img = img(:,:,1);
    end 
    for quality = quality_all
        fprintf('x%d ', quality);
        folder_LQ = fullfile(path_output_LQ, ['Q', num2str(quality)]);
        folder_HQ = fullfile(path_output_HQ, ['Q', num2str(quality)]);
        if ~exist(folder_LQ)
            mkdir(folder_LQ)
        end
        if ~exist(folder_HQ)
            mkdir(folder_HQ)
        end
        fn_LQ = fullfile(path_output_LQ, ['Q', num2str(quality)], [name_im(1:end-4), 'q', num2str(quality), '.png']);
        fn_HQ = fullfile(path_output_HQ, name_im);
        imwrite(img, fn_LQ, 'jpg', 'Quality', quality);
    imwrite(img, fn_HQ, 'png');
    end
    fprintf('\n');
end
fprintf('\n');
end















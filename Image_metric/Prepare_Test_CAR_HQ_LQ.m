function Prepare_Test_CAR_HQ_LQ()
clear all; close all; clc
path_original = '.\OriginalTestData';
dataset  = {'LIVE1','Classic5'};
ext = {'*.jpg', '*.png', '*.bmp'};

quality_all = [10, 20, 30, 40];

for idx_set = 1:length(dataset)
    fprintf('Processing %s:\n', dataset{idx_set});
    filepaths = [];
    for idx_ext = 1:length(ext)
        filepaths = cat(1, filepaths, dir(fullfile(path_original, dataset{idx_set}, ext{idx_ext})));
    end
    for idx_im = 1:length(filepaths)
        name_im = filepaths(idx_im).name;
        fprintf('%d. %s: ', idx_im, name_im);
        img = imread(fullfile(path_original, dataset{idx_set}, name_im));
        if size(img,3)>1
              img = rgb2gray(img);
        end 
        for quality = quality_all
            fprintf('x%d ', quality);
            folder_HQ = fullfile('.\HQ', dataset{idx_set}, ['Q', num2str(quality)]);
            folder_LQ = fullfile('.\LQ', dataset{idx_set}, ['Q', num2str(quality)]);
            if ~exist(folder_HQ)
                mkdir(folder_HQ)
            end
            if ~exist(folder_LQ)
                mkdir(folder_LQ)
            end
            % fn
            fn_HQ = fullfile('.\HQ', dataset{idx_set}, ['Q', num2str(quality)], [name_im(1:end-4), '_HQ_Q', num2str(quality), '.png']);
            fn_LQ = fullfile('.\LQ', dataset{idx_set}, ['Q', num2str(quality)], [name_im(1:end-4), '_LQ_Q', num2str(quality), '.png']);
            imwrite(img, fn_LQ, 'jpg', 'Quality', quality);
            imwrite(img, fn_HQ, 'png');
        end
        fprintf('\n');
    end
    fprintf('\n');
end
end

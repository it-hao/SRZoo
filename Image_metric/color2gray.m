function color2gray()
img = imread('bikes.bmp');
img = rgb2gray(img);
imwrite(img, 'bikes_gray.png');
end
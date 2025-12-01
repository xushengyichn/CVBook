clc
clear
close all

% 读取并显示封面图像
imgPath = fullfile('D:','Users','xushe','Documents','GitHub','CVBook','imgs','book_cover.jpg');

if ~exist(imgPath, 'file')
	fprintf('未找到文件: %s\n', imgPath);
	fprintf('请确认路径和文件名是否正确。\n');
	return;
end

I = imread(imgPath);
figure('Name','Book Cover','NumberTitle','off');
imshow(I);
axis image;
title('book_cover.jpg','Interpreter','none');


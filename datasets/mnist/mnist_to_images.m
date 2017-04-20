clear all;  close all;  clc;

addpath('mnistHelper');

%% Train set
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');

train_dir = 'train';
mkdir(train_dir);
fid = fopen('train.txt', 'w');
for i = 1 : numel(labels)
    img = reshape(images(:, i), [28 28]);
    lab = labels(i);
    img_path = [train_dir '/' num2str(i, '%05d') '.bmp'];
    imwrite(img, img_path);
    fprintf(fid, '%s %d\n', img_path, lab);
end
fclose(fid);

%% Test set
images = loadMNISTImages('t10k-images.idx3-ubyte');
labels = loadMNISTLabels('t10k-labels.idx1-ubyte');

test_dir = 'test';
mkdir(test_dir);
fid = fopen('test.txt', 'w');
for i = 1 : numel(labels)
    img = reshape(images(:, i), [28 28]);
    lab = labels(i);
    img_path = [test_dir '/' num2str(i, '%05d') '.bmp'];
    imwrite(img, img_path);
    fprintf(fid, '%s %d\n', img_path, lab);
end
fclose(fid);

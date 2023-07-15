% This code done in matlab 2019a to classify Acute lymphocytic leukemia
% images to blast or normal images
% designed by: Dr. Mohamed Maher Ata (mmaher844@yahoo.com), Eng. Reem Magdy Elrefaie (ramomagdy97@gmail.com)
warning('off')
clear 
close all
clc
%%%%%%%%%%%%%%%%%%%%choose your image%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

startingFolder = 'C:\Program Files\MATLAB';
if ~exist(startingFolder, 'dir')
  startingFolder = pwd;
end
defaultFileName = fullfile(startingFolder, '.');
[baseFileName, folder] = uigetfile(defaultFileName, 'Select a file');
if baseFileName == 0
  return;
end
fullFileName = fullfile(folder, baseFileName);
%%%%%%%%%%%5%%%%%%%%%%%%%%%start algorithm%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read the image
img=imread(fullFileName);

% Preprocessing step.... Data Augmentation
% Rotation...(you can choose any angle)
y=imrotate(img,90);
imshow(y)

% Translation..(you can try another number)
x=imtranslate(img,[15, 25]);
imshow(x)

% Segmentation...using k-means (this step applied on original images and
% images after rotation and traslation)

 
%  Step 1: Convert Image from RGB Color Space to L*a*b* Color Space
cform = makecform('srgb2lab');
lab_he = applycform(img,cform);
% imshow(lab_he),title(lab)

% Step 2: Classify the Colors in 'a*b*' Space Using K-Means Clustering
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);

nColors = 3;
% repeat the clustering 3 times to avoid local minima
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3 );
                                     
% Step 3: Label Every Pixel in the Image Using the Results from KMEANS
pixel_labels = reshape(cluster_idx,nrows,ncols);
figure,imshow(pixel_labels,[]), title('image labeled by cluster index');


% Step 4: Create Images that Segment the H&E Image by Color.
segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels,[1 1 3]);

for k = 1:nColors
    color = img;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color;
end

figure,imshow(segmented_images{1}), title('objects in cluster 1');
% 
figure,imshow(segmented_images{2}), title('objects in cluster 2');
% 
figure,imshow(segmented_images{3}), title('objects in cluster 3');


% Step 5: Segment the Nuclei into a Separate Image
mean_cluster_value = mean(cluster_center,2);
[tmp, idx] = sort(mean_cluster_value);
blue_cluster_num = idx(1);

L = lab_he(:,:,1);
blue_idx = find(pixel_labels == blue_cluster_num);
L_blue = L(blue_idx);
is_light_blue = imbinarize(L_blue);

nuclei_labels = repmat(uint8(0),[nrows ncols]);
nuclei_labels(blue_idx(is_light_blue==false)) = 1;
nuclei_labels = repmat(nuclei_labels,[1 1 3]);
blue_nuclei = img;
blue_nuclei(nuclei_labels ~= 1) = 0;
imshow(blue_nuclei),title("segmented image");

% Feature extraction step( we use nine different methods...Applied this step in all images in dataset to create excel sheet)
% put image after segmentation....


F = uint8(255 *blue_nuclei );
F=im2double((rgb2gray(F)));
% 1) PCA:
coeff = pca(F);
coeff=mean(coeff);

% 2) ICA:
Mdl = rica(F,10);
z = transform(Mdl,F);
z=var((z));

% 3) GLCM
glcms = graycomatrix(F);
glcms=var((glcms));

% 4) LBP
features = extractLBPFeatures(F);

% 5) DCT
dc = dct2(F);
dc=var((dc));

% 6) DFT
D = fft(F);
D=var((D));

% 7) DWT
[cA,cH,cV,cD] = dwt2(F,'sym4','mode','per');
cH=var((cH));

% 8) shape
s = regionprops('table',F,'Area','Perimeter','Solidity','Circularity','Eccentricity','ConvexArea','MajorAxisLength','MinorAxisLength');

% 9) EMD
F=var(F);
rParabEmd= rParabEmd__L(F,40, 40, 1 );
rParabEmd=var(rParabEmd);
% using rParabEmd_l.m file

% Classification step...(using excel sheet created from feature extraction
% step)
% Solve a Pattern Recognition Problem with a Neural Network


% This script assumes these variables are defined:
%
%   ALLInputs - input data.
%   ALLTargets - target data.
data = readmatrix('C:\Users\Desktop\EMDfeatures.csv');
c = data(:,1:9);
y = data(:,10);
x = c';
t = y';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainbr';  % Scaled conjugate gradient backpropagation.

% Create a Pattern Recognition Network
hiddenLayerSize = 15;
net = patternnet(hiddenLayerSize, trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivision
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'crossentropy';  % Cross-Entropy

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y)
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

% Recalculate Training, Validation and Test Performance
trainTargets = t .* tr.trainMask{1};
valTargets = t .* tr.valMask{1};
testTargets = t .* tr.testMask{1};
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)

% View the Network
view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotconfusion(t,y)
%figure, plotroc(t,y)

% Deployment
% Change the (false) values to (true) to enable the following code blocks.
% See the help for each generation function for more information.
if (false)
    % Generate MATLAB function for neural network for application
    % deployment in MATLAB scripts or with MATLAB Compiler and Builder
    % tools, or simply to examine the calculations your trained neural
    % network performs.
    genFunction(net,'myNeuralNetworkFunction');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a matrix-only MATLAB function for neural network code
    % generation with MATLAB Coder tools.
    genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
    y = myNeuralNetworkFunction(x);
end
if (false)
    % Generate a Simulink diagram for simulation or deployment with.
    % Simulink Coder tools.
    gensim(net);
end

% end of our steps

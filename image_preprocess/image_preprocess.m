%% Test
clear all
folderNames = [
% "210803_Suwon_Ginkgo";
% "210803_Suwon_Zelkova";
% "210811_Suwon_Zelkova";
% "210811_Suwon_Prunus";
% "210811_Suwon_Mixed";
% "210826_Suwon_Mixed1";
% "210826_Suwon_Mixed2";
% "210826_Suwon_Mixed3";
% "210830_Suwon_Mixed1";
% "210910_Suwon_Mixed1";
% "210924_Suwon_Mixed1";
% "210924_Suwon_Mixed2";
% "210924_Suwon_Mixed3";
% "210924_Suwon_Mixed4";
% "210927_Suwon_Mixed1";
% "210927_Suwon_Mixed2";
% "210927_Suwon_Mixed3";
% "211001_Suwon_Mixed1";
% "211001_Suwon_Mixed2";
% "211001_Suwon_Mixed4";
% "211001_Suwon_Mixed5";
% "211013_Suwon_Mixed1";
% "211013_Suwon_Mixed2";
% "211013_Suwon_Mixed3";
% "211013_Suwon_Mixed4";
% "211021_Suwon_Mixed1";
% "211021_Suwon_Mixed2";
"2024-08-28-12-49-28_SNU";
]

folderNum = length(folderNames);

parfor g = 1:folderNum
    folderName = folderNames(g)

    inputDir= strcat("/esail3/Tackang/97CarsenseData/",folderName);
    inputDir =  strcat(inputDir,"/preprocessed_data/image");
    % inputDir = "/bess23/Tackang/CarsenseData/210803_Suwon_Zelkova/KITTI_Final/image_00_cal";
    % outputDir = "/bess23/Tackang/CarsenseData/210803_Suwon_Zelkova/KITTI_Final/image_00_cal_processed";
    outputDir = strcat(inputDir,"_processed/");
    check = exist(outputDir,'dir');
    if check ==0
        mkdir(outputDir);
    end
    system(sprintf('chmod 777 %s',outputDir));
    cd(inputDir)

    img_list = dir("*.jpg");
    num = length(img_list);

    for i = 1:num
        cd(outputDir)
        File = strcat(outputDir,img_list(i).name)
        checkFile = exist(File,'file');
        if checkFile ~= 0
            continue;
        end
        
        if img_list(i).bytes == 0
            image = zeros(1200,1920,3);
            image(image == 0) = 255;
            cd(outputDir)
            imwrite(image,img_list(i).name)
            continue;
        end
        
        cd(inputDir)
        
        threshold = 0.5;
        img = imread(img_list(i).name);
        img_resized = imresize(img, 0.1, 'nearest');

        img_LAB = rgb2lab(img_resized);
        L = img_LAB(:,:,1)/100;
        brightness = mean(mean(L),2);

        if brightness >= threshold
            I1_R = img(:,:,1);
            I1_G = img(:,:,2);
            I1_B = img(:,:,3);

            I1_R_Adj = imadjust(I1_R);
            I1_G_Adj = imadjust(I1_G);
            I1_B_Adj = imadjust(I1_B);

            I1_dimension = size(img);
            I1_new = zeros(I1_dimension(1), I1_dimension(2), I1_dimension(3));
            I1_new(:,:,1) = I1_R_Adj;
            I1_new(:,:,2) = I1_G_Adj;
            I1_new(:,:,3) = I1_B_Adj;
            I1_new = uint8(I1_new);

            I1_new_brighten = imlocalbrighten(I1_new, 0.2);

            sigma = 0.2;
            alpha = 0.3;
            I1_Laplacian = locallapfilt(I1_new_brighten, sigma, alpha, 'NumIntensityLevels', 10);

            cd(outputDir)
            imwrite(I1_Laplacian, img_list(i).name)

        else
            I2_R = img(:,:,1);
            I2_G = img(:,:,2);
            I2_B = img(:,:,3);

            I2_R_Eq = histeq(I2_R);
            I2_G_Eq = histeq(I2_G);
            I2_B_Eq = histeq(I2_B);

            I2_dimension = size(img);
            I2_new = zeros(I2_dimension(1), I2_dimension(2), I2_dimension(3));
            I2_new(:,:,1) = I2_R_Eq;
            I2_new(:,:,2) = I2_G_Eq;
            I2_new(:,:,3) = I2_B_Eq;
            I2_new = uint8(I2_new);

            I2_LAB = rgb2lab(I2_new);
            I2_L = I2_LAB(:,:,1)/100;
            I2_L = adapthisteq(I2_L, 'NumTiles', [8 8], 'ClipLimit', 0.005);
            I2_LAB(:,:,1) = I2_L * 100;
            I2_J = lab2rgb(I2_LAB);
            I2_J = im2uint8(I2_J);

            I2_new_brighten = imlocalbrighten(I2_J, 0.25);
            I2_new_dehazed= imreducehaze(I2_new_brighten, 0.2);

            cd(outputDir)
            imwrite(I2_new_dehazed, img_list(i).name)
        end
   
    end
    

end


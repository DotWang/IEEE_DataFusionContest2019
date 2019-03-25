
clear
%%
path='N:\RS\Data Fusion_jiqun\TRACK1\PostProcessing\Submission5\';
path_idx='N:\RS\Data Fusion_jiqun\TRACK1\PostProcessing\Test_Index\';
path_fix='';
%%
CLS = imread(strcat(path,'JAX_160_001_CLS.tif')); 
DSM = imread(strcat(path,'JAX_160_001_AGL.tif'));
NDVI= imread(strcat(path_idx,'JAX_160_001_NDVI_split.tif'));
NDVI=NDVI>1;
figure
imshow(ColorMap(CLS))
title('Origin')
%% catagory config
fix_Veg=0;
fix_Bui=1;
fix_Roa=1;
%% fix the Vegetation
if fix_Veg
    CLS_highVe = CLS==5;
    CLS_highVe_candidate = NDVI==1 & DSM>2.5 ;
    CLS_highVe_correct = CLS;
    CLS_highVe_correct(find(CLS_highVe_candidate>0)) = 5;

    % fix hole
    CLS_highVe_hole = CLS_highVe & NDVI==0 & DSM<2.5;
    CLS_highVe_correct(find(CLS_highVe_hole>0)) = 2;

    CLS=CLS_highVe_correct;
    figure
    imshow(ColorMap(CLS))
    title('Fixed Vegetation!')
end
%% fix the building
if fix_Bui
    CLS_building = CLS==6;
    CLS_building = imfill(double(CLS_building),'holes');% fix hole

    CLS_building_correct = CLS;
    
    CLS_building_hole = CLS_building & DSM<2;
    CLS_building_correct(find(CLS_building_hole>0)) = 2;
    
    CLS=CLS_building_correct;
    figure
    imshow(ColorMap(CLS))
    title('Fixed Building!')
end
%% fix the road
if fix_Roa
    CLS_bridge = CLS==17;
    CLS_bridge=imfill(double(CLS_bridge),'holes');% fix hole
    
    CLS_bridge_Dilate = imdilate(CLS_bridge,strel('disk',10));
    CLS_bridge_candidate = CLS_bridge_Dilate & DSM>2;
    CLS_bridge_correct = CLS;
    CLS_bridge_correct(find(CLS_bridge_candidate>0)) = 17;

    % fix hole between road
    CLS_bridge_hole = CLS_bridge & DSM<2;
    CLS_bridge_correct(find(CLS_bridge_hole>0)) = 2;
    
    CLS=CLS_bridge_correct;
    figure
    imshow(ColorMap(CLS))
    title('Fixed Road!')
end

%imwrite(CLS,strcat(path_fix,'JAX_160_001_CLS.tif'))












clear
clc
tic
fileFolder=fullfile('N:\RS\Data Fusion_jiqun\TRACK1\PostProcessing\Test_Index\');

dirOutput=dir(fullfile(fileFolder,'*.tif'));

fileNames={dirOutput.name}';

for i=1:100
    name_image=char(strcat('N:\RS\Data Fusion_jiqun\TRACK1\PostProcessing\Test_Index\',string(fileNames{i})));
    
    R=regexp(name_image,'\.','split');%split the string of filename
    r1=char(R(1));%extract_index
    
    disp(name_image)
    img_matrix=imread(name_image);%index matrix & geospatial Reference
    img_matrix=double(img_matrix);
    for j=1:size(img_matrix,1)%shrink the range of matrix values
        for k=1:size(img_matrix,2)
            if img_matrix(j,k)<-5
                img_matrix(j,k)=-5;
            end
            if img_matrix(j,k)>5
                img_matrix(j,k)=5;
            end
        end
    end
    identify=isnan(img_matrix);%find the nan_values
    img_matrix=mat2gray(img_matrix);%scale to 0-1 because of negative number
    img_matrix=im2uint8(img_matrix);%scale to 0-255
    img_matrix=double(img_matrix);
    img_matrix_truth=img_matrix;%create another img_matrix existing nan
    for j=1:size(identify,1)%shrink the range of matrix values
        for k=1:size(identify,2)
            if identify(j,k)==1
                img_matrix(j,k)=0;%make nan to 0
            end
        end
    end
    for j=1:size(identify,1)%shrink the range of matrix values
        for k=1:size(identify,2)
            if identify(j,k)==1
                img_matrix_truth(j,k)=nan;%make nan still nan
            end
        end
    end
    Cov_in=zeros(256,1);%within-class scatter matrix
    Cov_out=zeros(256,1);%Between class scatter matrix
    signal=zeros(256,1);%matrix for certain the t
    S=regexp(r1,'\_','split');%split the string of filename
    s6=char(S(6));%extract_index
    for t=0:255
        count_1=0;count_2=0;%count of one two parts
        sum_1=0;sum_2=0;%sum all values of one of two parts
        mean_1=-1;mean_2=1;%average of one of two parts
        Variance_1=-1;Variance_2=-1;
        %Variance of of one of two parts
        %% average value
        for j=1:size(img_matrix,1)
            for k=1:size(img_matrix,2)
                if isnan(img_matrix_truth(j,k))==1
                    continue;
                else
                    if img_matrix_truth(j,k)<=t
                        count_1=count_1+1;
                        sum_1=sum_1+img_matrix(j,k);
                    else
                        count_2=count_2+1;
                        sum_2=sum_2+img_matrix(j,k);
                    end
                end
            end
        end
        if count_1==0
            mean_1=0;
        else
            mean_1=sum_1*1.0./count_1;%average value of part 1
        end
        if count_2==0
            mean_2=0;
        else
            mean_2=sum_2*1.0./count_2;%average value of part 2
        end
        %% Variance
        sumvar_1=0;sumvar_2=0;
        for j=1:size(img_matrix,1)
            for k=1:size(img_matrix,2)
                if isnan(img_matrix_truth(j,k))==1
                    continue;
                else
                    if img_matrix_truth(j,k)<=t
                        sumvar_1=sumvar_1+(img_matrix(j,k)-mean_1)^2;
                    else
                        sumvar_2=sumvar_2+(img_matrix(j,k)-mean_2)^2;
                    end
                end
            end
        end
        if count_1==0
            Variance_1=0;
        else
            Variance_1=sumvar_1*1.0./count_1;%variance of part 1
        end
        if count_2==0
            Variance_2=0;
        else
            Variance_2=sumvar_2*1.0./count_2;%variance of part 2
        end
        mean_all=(count_1*mean_1+count_2*mean_2)*1.0./(count_1+count_2);
        Cov_in(t+1,1)=count_1*Variance_1+count_2*Variance_2;
        Cov_out(t+1,1)=count_1*(mean_1-mean_all)^2+count_2*(mean_2-mean_all)^2;
        signal(t+1,1)=Cov_out(t+1)./Cov_in(t+1);
    end
    x=signal(1);
    number=1;
    for h=1:256%choose the largest value
        if signal(h)>x
            x=signal(h);
            number=h-1;
        end
    end
    img_matrix_new=-zeros(size(img_matrix,1),size(img_matrix,2));%initialize 0
    if strcmp(s6,'NDWI')==1 | strcmp(s6,'NDVI')==1
        for j=1:size(img_matrix,1)
            for k=1:size(img_matrix,2)
                if isnan(img_matrix_truth(j,k))==1
                    continue;
                else
                    if img_matrix(j,k)>number
                        img_matrix_new(j,k)=1;
                    else
                        img_matrix_new(j,k)=0;
                    end
                end
            end
        end
    else 
        for j=1:size(img_matrix,1)
            for k=1:size(img_matrix,2)
                if isnan(img_matrix_truth(j,k))==1
                    continue;
                else
                    if img_matrix(j,k)<number
                        img_matrix_new(j,k)=1;
                    else
                        img_matrix_new(j,k)=0;
                    end
                end
            end
        end
    end
    
    result=strcat(r1,'_Split.tif');
    imwrite(img_matrix_new,result)
end
fprintf('Finsihed\n');
toc

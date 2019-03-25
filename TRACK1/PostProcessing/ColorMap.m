function Map=ColorMap(label)
    pallet = hsv(5);
    [row,col] = size(label);
    Map = zeros(row*col,3);
    label = label(:);
    gt = [2,5,6,9,17];
    for i=1:5
        Map(label==gt(i),1) = pallet(i,1);
        Map(label==gt(i),2) = pallet(i,2);
        Map(label==gt(i),3) = pallet(i,3);
    end
    Map = reshape(Map,row,col,3);
end
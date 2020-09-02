min_ent = 100;
min_i = 0;
min_j = 0;

for i = 0:207
    for j = 0:207
        i_str = int2str(i);
        j_str = int2str(j);
        filename = [i_str ',' j_str '.png'];
        img = imread(filename);
        ent = entropy(img);
        if i == 88 && j == 44
            ent
        end
        if ent < min_ent
            min_ent = ent;
            min_i = i;
            min_j = j;
        end
    end
end

min_ent
min_i
min_j

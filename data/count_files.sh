#!/bin/bash

default_path_one="1789to1824_DebatesAndProceedings"
default_path_two="1824to1837_DebatesAndProceedings"

dir_one='volumes'
dir_two='items'

subdir_one='Volume_'
subdir_two='Item_'

end_one=42
end_two=29

default_path=${default_path_one}
dir=${dir_one}
subdir=${subdir_one}
end=${end_one}


echo "Main path: ${default_path}"
for i in `seq 1 1 ${end}`; do

  c=$(find ${default_path}/${dir}/${subdir}${i} -type f -name 'page*' | wc -l)
  echo "Pages in /${dir}/${subdir}${i}: ${c}"
  sum=$((sum+c))

  c=$(find ${default_path}/hocrs_files/${subdir}${i} -type f -name 'page*' | wc -l)
  echo "Pages in /hocrs_files/${subdir}${i}: ${c}"
  sum=$((sum+c))

  c=$(find ${default_path}/one_column_oriented/${subdir}${i} -type f -name 'page*' | wc -l)
  echo "Pages in /one_column_oriented/${subdir}${i}: ${c}"
  sum=$((sum+c))

  c=$(find ${default_path}/text_volumes/${subdir}${i} -type f -name 'page*' | wc -l)
  echo "Pages in /text_volumes/${subdir}${i}: ${c}, ${c}x4: $(($c * 4))"
  sum=$((sum+c))

  c=$(find ${default_path}/speeches/${subdir}${i} -type f -name 'page*' | wc -l)
  echo "Pages in /speeches/${subdir}${i}: ${c}"
  sum=$((sum+c))

  c=$(find ${default_path}/df_tuples/${subdir}${i} -type f -name 'tuples_counts.csv' | wc -l)
  echo "Pages in /df_tuples/${subdir}${i}: ${c}"
  sum=$((sum+c))

  echo ""

done


echo $sum

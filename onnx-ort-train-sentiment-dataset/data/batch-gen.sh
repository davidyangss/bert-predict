#!/bin/bash -l

script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# // for 100 files
# //   for 8000 line

for i in {1..100}; do
  file="$script_path/../data/csvs/file_$i.txt"
  echo "id,comment,sentiment" > $file
  for j in {1..200000}; do
    echo "$(expr $i \* $j),涉及党内有争议尚未做出结论的重大问题以及重大政治历史事件不宜公开的档案、材料；涉及地方重大事件不宜公开的档案、材料。,0" >> $file
  done
done
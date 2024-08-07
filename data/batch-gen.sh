#!/bin/bash -l

script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# // for 100 files
# //   for 8000 line

for i in {1..100}; do
  file="$script_path/csvs/file_$i.txt"
  echo "id,text,label" > $file
  for j in {1..200000}; do
    echo "$(expr $i \* $j),5月8日付款成功，当当网显示5月10日发货，可是至今还没看到货物，也没收到任何通知，简不知怎么说好！！！,0" >> $file
  done
done
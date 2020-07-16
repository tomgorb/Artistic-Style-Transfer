while read file;
do 
  convert dataset/$file -resize 256x256! training/$file
done < <(ls dataset/ | awk '{print $1}')

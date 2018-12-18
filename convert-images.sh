
# safety first
set -e -o pipefail
shopt -s failglob 

# resize to 299x299, but keep image size
# drop -resize if zooming with crop is better (but maybe not?)

BASE=ZooScanSet/imgs
OUT=data

ls "$BASE" | tail -n +14 | while read dir; do
   echo "$dir"
   mkdir -p "$OUT/$dir"
   ls "$BASE/$dir" | while read f; do 
          convert -resize 299x299 "$BASE/$dir/$f" -background white -gravity center -extent 299x299 "$OUT/$dir/$f"
   done
done


ls data | while read x; do
    printf "%-32s	%8d\n" "$x" $(ls "data/$x" | wc -l)
done

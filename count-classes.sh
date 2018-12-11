
for x in $(ls data); do
    echo "$x	$(ls data/$x | wc -l)"
done

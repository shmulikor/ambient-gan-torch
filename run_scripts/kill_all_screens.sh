str=$(screen -ls)

array=$(echo $str|tr "." "\n")

for V in $array
do
if [ $V -gt 0  ]
    then screen -S $V -X quit
fi
done

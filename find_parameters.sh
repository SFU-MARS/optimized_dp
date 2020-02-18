#!/bin/bash

for k in 1 2 4 5 10
do
    for j in 1 2 4 5 10
      do
	  for i in 1 2 4 5 10
	    do	  
	     python3 para_opt_HeteroCL.py -i $i -j $j -k $k >> para_opt_res.txt
	    done
      done	 
done

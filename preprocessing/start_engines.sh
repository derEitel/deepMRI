#!/bin/bash

END=$1
for i in $(seq 1 $END);
do
   nohup ipengine --profile='grid' &
done

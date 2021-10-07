#!/bin/sh

path_to_darknet="$1"
"$path_to_darknet" "detector" "train" "./data/swarming_yolo/labelled_data.data" "./models/yolo/cfg/darknet-yolo4.cfg" "./models/yolo/darknet/darknet53_v4.conv.74" "-dont_show"

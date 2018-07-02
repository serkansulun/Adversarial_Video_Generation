#!/bin/bash
dataset="KTH"
output_main_folder="${dataset}_frames"
mkdir $output_main_folder
for category in $dataset/*/ ; do
	category=${category%*/}
	category=${category##*/}
	mkdir $output_main_folder/$category
    for video in $dataset/$category/*.avi ; do
    	video_m=${video%*/}
		video_m=${video_m##*/}
    	mkdir $output_main_folder/$category/$video_m
		ffmpeg -i $video -r 25 $output_main_folder/$category/$video_m/$filename%04d.png
	done

done
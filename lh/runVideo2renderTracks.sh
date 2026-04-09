#!/bin/bash


cd ..

dataType="seva2som" # seva2som,ac3d2som
nameSeq="lily-dragon_orbit" # blue-car_lemniscate
nameFolder="maskAllOneCls_py" # lily-dragon,lemniscate,blue-car
nameTrain="${nameSeq}_${nameFolder}"
dirData="${dataType}/${nameFolder}"


echo "[${nameSeq}] start run_video.py ..."
python run_video.py \
    --work_dir /root/autodl-tmp/som-change/lh/${nameTrain} \
    data:custom --data.data_dir  /root/autodl-tmp/data/${dirData} --data.camera-type droid_recon --data.seq_name ${nameSeq} \
    trajectory:wander \
    time:replay
echo "${nameSeq} finish run_video.py ..."
sleep 3
echo


echo "start render_tracks.py ..."
python render_tracks.py \
    --work_dir /root/autodl-tmp/shape-of-motion/lh/${nameTrain} \
    --no-use_2dgs \
    data:custom --data.data_dir /root/autodl-tmp/data/${dirData} --data.camera-type droid_recon --data.seq_name ${nameSeq} \
    trajectory:wander\
    time:replay
echo "${nameSeq} finish render_tracks.py"

#!/bin/bash


cd ..


# paper-windmill
# python run_video.py \
#     --work_dir /root/autodl-tmp/shape-of-motion/lh/paper-windmill \
#     data:iphone --data.data_dir  /root/autodl-tmp/data/paper-windmill \
#     trajectory:lemniscate \
#     time:replay

# custom_2dgs;cat
# python run_video.py \
#     --work_dir /root/autodl-tmp/shape-of-motion/lh/custom_2dgs \
#     data:custom --data.data_dir  /root/autodl-tmp/data/seva2som --data.camera-type droid_recon --data.seq_name custom_2dgs \
#     trajectory:lemniscate \
#     time:replay


# # custom_2dgs;cat
# python run_video.py \
#     --work_dir /root/autodl-tmp/shape-of-motion/lh/custom \
#     data:custom --data.data_dir  /root/autodl-tmp/data/seva2som --data.camera-type droid_recon --data.seq_name cat \
#     trajectory:lemniscate \
#     time:replay


# # lemniscate: 双纽线；spiral：螺旋的；wander：漫步；fixed：固定
# name="sika"
# dirData="ac3d2som/sika" # seva2som
# python run_video.py \
#     --work_dir /root/autodl-tmp/shape-of-motion/lh/sika \
#     data:custom --data.data_dir  /root/autodl-tmp/data/${dirData} --data.camera-type droid_recon --data.seq_name ${name} \
#     trajectory:wander \
#     time:replay

# name="garden_flythrough"
# dirData="seva2som/${name}" # seva2som
# python run_video.py \
#     --work_dir /root/autodl-tmp/shape-of-motion/lh/${name} \
#     data:custom --data.data_dir  /root/autodl-tmp/data/${dirData} --data.camera-type droid_recon --data.seq_name ${name} \
#     trajectory:wander \
#     time:replay


# # bear,retriever1,retriever2,sika1,sika2,sheep
# name="sika2"
# cls="ac3d2som" # seva2som,ac3d2som
# dirData="${cls}/select" # seva2som,ac3d2som
# python run_video.py \
#     --work_dir /root/autodl-tmp/shape-of-motion/lh/${name} \
#     data:custom --data.data_dir  /root/autodl-tmp/data/${dirData} --data.camera-type droid_recon --data.seq_name ${name} \
#     trajectory:wander \
#     time:replay


# lily-dragon_lemniscate;lily-dragon_move-backward,,lily-dragon_move-right,lily-dragon_orbit,lily-dragon_zoom-out
# vasedeck_lemniscate
# blue-car_lemniscate
dataType="seva2som" # seva2som,ac3d2som
nameSeq="lily-dragon_orbit" # blue-car_lemniscate
nameFolder="maskAllOneCls_py" # lily-dragon,lemniscate,blue-car
nameTrain="${nameSeq}_${nameFolder}"
dirData="${dataType}/${nameFolder}"
python run_video.py \
    --work_dir /root/autodl-tmp/som-change/lh/${nameTrain} \
    data:custom --data.data_dir  /root/autodl-tmp/data/${dirData} --data.camera-type droid_recon --data.seq_name ${nameSeq} \
    trajectory:wander \
    time:replay
   


    
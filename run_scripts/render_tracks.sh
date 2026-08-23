cd ..

# no-skip_load_imgs: skip_load_imgs=False;no-use_2dgs: use_2dgs=False
# iphone,backpack,paper-windmill；spin
# dataName="backpack"
# out="lh50"
# python render_tracks.py \
#     --work_dir /root/autodl-tmp/som-change/${out}/${dataName} \
#     --no-use_2dgs \
#     data:iphone --data.data_dir  /root/autodl-tmp/data/${dataName} --data.no-skip_load_imgs \
#     trajectory:lemniscate \
#     time:replay

# # custom
# python render_tracks.py \
#     --work_dir /root/autodl-tmp/shape-of-motion/results/custom_2dgs \
#     --no-use_2dgs \
#     data:custom --data.data_dir /root/autodl-tmp/data/seva2som --data.camera-type droid_recon --data.seq_name garden_flythrough_seva_16 \
#     trajectory:lemniscate \
#     time:replay


# # lemniscate
# name="sika"
# python render_tracks.py \
#     --work_dir /root/autodl-tmp/shape-of-motion/results/${name} \
#     --no-use_2dgs \
#     data:custom --data.data_dir /root/autodl-tmp/data/ac3d2som/${name} --data.camera-type droid_recon --data.seq_name ${name} \
#     trajectory:wander\
#     time:replay

# # lemniscate
# name="sheep" # garden_flythrough
# dirData="ac3d2som/select" #seva2som,ac3d2som
# python render_tracks.py \
#     --work_dir /root/autodl-tmp/shape-of-motion/results/${name} \
#     --no-use_2dgs \
#     data:custom --data.data_dir /root/autodl-tmp/data/${dirData} --data.camera-type droid_recon --data.seq_name ${name} \
#     trajectory:wander\
#     time:replay


# retriever2,sheep,sika1; | # lily-dragon_lemniscate;lily-dragon_move-backward,,lily-dragon_move-right,lily-dragon_orbit,lily-dragon_zoom-out
# vasedeck_lemniscate
# blue-car_lemniscate,blue-car_orbit
dataType="ac3d2som" # seva2som,ac3d2som,maskAll 
nameFolder="maskfg_select250625" # maskfg_select250625,maskAll_select250625
out="results"
name="tiger2_00018" # tiger2_00018,astronaut_00005
nameTrain="${name}_${nameFolder}_any4dv370-b18"
# nameTrain="${name}_${nameFolder}"
dirData="${dataType}/${nameFolder}"
python render_tracks.py \
    --work_dir /root/autodl-tmp/som-change/${out}/${nameTrain} \
    --no-use_2dgs \
    data:custom --data.data_dir /root/autodl-tmp/data/${dirData} --data.camera-type droid_recon --data.seq_name ${name} \
    trajectory:wander\
    time:replay
    
    
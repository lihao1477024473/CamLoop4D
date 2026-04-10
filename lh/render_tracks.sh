cd ..

# no-skip_load_imgs: skip_load_imgs=False;no-use_2dgs: use_2dgs=False
# iphone,backpack,paper-windmill；spin
dataName="paper-windmill"
python render_tracks.py \
    --work_dir /root/autodl-tmp/som-change/lh/${dataName} \
    --no-use_2dgs \
    data:iphone --data.data_dir  /root/autodl-tmp/data/${dataName} --data.no-skip_load_imgs \
    trajectory:lemniscate \
    time:replay

# # custom
# python render_tracks.py \
#     --work_dir /root/autodl-tmp/shape-of-motion/lh/custom_2dgs \
#     --no-use_2dgs \
#     data:custom --data.data_dir /root/autodl-tmp/data/seva2som --data.camera-type droid_recon --data.seq_name garden_flythrough_seva_16 \
#     trajectory:lemniscate \
#     time:replay


# # lemniscate
# name="sika"
# python render_tracks.py \
#     --work_dir /root/autodl-tmp/shape-of-motion/lh/${name} \
#     --no-use_2dgs \
#     data:custom --data.data_dir /root/autodl-tmp/data/ac3d2som/${name} --data.camera-type droid_recon --data.seq_name ${name} \
#     trajectory:wander\
#     time:replay

# # lemniscate
# name="sheep" # garden_flythrough
# dirData="ac3d2som/select" #seva2som,ac3d2som
# python render_tracks.py \
#     --work_dir /root/autodl-tmp/shape-of-motion/lh/${name} \
#     --no-use_2dgs \
#     data:custom --data.data_dir /root/autodl-tmp/data/${dirData} --data.camera-type droid_recon --data.seq_name ${name} \
#     trajectory:wander\
#     time:replay


# retriever2,sheep,sika1; | # lily-dragon_lemniscate;lily-dragon_move-backward,,lily-dragon_move-right,lily-dragon_orbit,lily-dragon_zoom-out
# vasedeck_lemniscate
# blue-car_lemniscate,blue-car_orbit
# dataType="ac3d2som" # seva2som,ac3d2som,maskAll 
# nameFolder="maskfg_select250625" 
# name="tiger1_00007"
# nameTrain="${name}_${nameFolder}_change_sample10_mofBase12_repeat_useSomCoef_numBases25"
# # nameTrain="${name}_${nameFolder}"
# dirData="${dataType}/${nameFolder}"
# python render_tracks.py \
#     --work_dir /root/autodl-tmp/som-change/lh/${nameTrain} \
#     --no-use_2dgs \
#     data:custom --data.data_dir /root/autodl-tmp/data/${dirData} --data.camera-type droid_recon --data.seq_name ${name} \
#     trajectory:wander\
#     time:replay
    
    
# https://github.com/vye16/shape-of-motion/issues/22

cd ..

# # custom;paper-windmill,backpack
# model="som-change"
# out="lh50"
# python run_rendering.py \
#     --work-dir /root/autodl-tmp/${model}/${out}/backpack/ \
#     --port 8091

# # custom
# python run_rendering.py \
#     --work-dir /root/autodl-tmp/shape-of-motion/lh/custom_2dgs/ \
#     --port 8091

# # sika
# python run_rendering.py \
#     --work-dir /root/autodl-tmp/shape-of-motion/lh/sika/ \
#     --port 8091

# bear,retriever1,retriever2,sika1,sika2,sheep;| lily-dragon_lemniscate,lily-dragon_move-backward,lily-dragon_move-right,lily-dragon_orbit,lily-dragon_zoom-out
# blue-car_lemniscate
# vasedeck_lemniscate,vasedeck_lemniscate-copy1

# retriever1,
# astronaut_00005_maskfg_select250625_change_sample10_mofBase12_repeat_useSomCoef_numBases25
# tiger1_00007_maskfg_select250625_change_sample10_mofBase12_repeat_useSomCoef_numBases25
model="som-change"
version="any4dv350"
name="astronaut_00005_maskAll_select250625_${version}-b18" # astronaut_00005,tiger2_00018
python run_rendering.py \
    --work-dir /root/autodl-tmp/${model}/lh/${name}/ \
    --port 8091

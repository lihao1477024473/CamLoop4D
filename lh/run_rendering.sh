# https://github.com/vye16/shape-of-motion/issues/22

cd ..

# # custom;paper-windmill
# python run_rendering.py \
#     --work-dir /root/autodl-tmp/shape-of-motion/lh/paper-windmill/ \
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
name="tiger1_00007_maskfg_select250625_change_sample10_mofBase12_repeat_useSomCoef_numBases25" # bear_maskAll;bear_maskAll_2dgs;retriever1_maskAll;bear_maskAllOneCls_py

python run_rendering.py \
    --work-dir /root/autodl-tmp/som-change/lh/${name}/ \
    --port 8091

cd ..

# python preproc/mask_app.py \
#     --root_dir /root/autodl-tmp/data/seva2som/garden_flythrough \
#     --checkpoint_dir /root/autodl-tmp/shape-of-motion/preproc/checkpoints

# python preproc/mask_app.py \
#     --root_dir /root/autodl-tmp/data/ac3d2som/sika \
#     --checkpoint_dir /root/autodl-tmp/shape-of-motion/preproc/checkpoints

# not finish
# # name="sheep"
# # dataDir="ac3d2som/${name}"
# dataDir="ac3d2som/select"
# python preproc/mask_app.py \
#     --root_dir /root/autodl-tmp/data/${dataDir} \
#     --checkpoint_dir /root/autodl-tmp/shape-of-motion/preproc/checkpoints


# lily-dragon,blue-car;lily-dragon-maskAll
dataType="ac3d2som"
name="sheep" # maskAll,maskAllOneCls;maskAllOneCls_py
dataDir="${dataType}/${name}"
python preproc/mask_app.py \
    --root_dir /root/autodl-tmp/data/${dataDir} \
    --checkpoint_dir /root/autodl-tmp/shape-of-motion/preproc/checkpoints


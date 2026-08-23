cd ..
cd preproc


# data_root='/root/autodl-tmp/data/seva2som'
# data_root='/root/autodl-tmp/data/ac3d2som/sika'
# data_root='/root/autodl-tmp/data/seva2som/garden_flythrough'

# name="sheep"
# dataDir="ac3d2som/${name}"
# python process_custom.py \
#     --img-dirs /root/autodl-tmp/data/${dataDir}/images/** --gpus 0


# name="select"
# dataDir="ac3d2som/${name}"
# python process_custom.py \
#     --img-dirs /root/autodl-tmp/data/${dataDir}/images/** --gpus 1

# lily-dragon,blue-car,vasedeck;lily-dragon-maskAll
dataType="seva2som" # seva2som
name="maskAllOneCls_py" # maskAllOneCls,maskAll
dataDir="${dataType}/${name}"
python process_custom.py \
    --img-dirs /root/autodl-tmp/data/${dataDir}/images/** --gpus 0
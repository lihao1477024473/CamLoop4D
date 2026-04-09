cd ..

# python train_launcher.py \
#     --devices 0 1 \
#     --seqs bear dog horse \
#     --work-root ./experiments/davis_depth \
#     --davis-root /data/DAVIS \
#     --res 480p \
#     --depth-type aligned_depth_anything

# --devices 0 1 \
model="som-change"
dataName="blackswan" 
python launch_davis.py \
    --devices 0 \
    --seqs ${dataName} \
    --work-root "/root/autodl-tmp/${model}/lh" \
    --davis-root /root/autodl-tmp/data/DAVIS \
    --res 480p \
    --depth-type aligned_depth_anything

# python visualize_and_save.py \
#     --data iphone \
#     --save \
#     --output-dir ./output_iphone

cd ..

name="sika"
# python vis_depths.py data:custom --data.data_dir /root/autodl-tmp/data/ac3d2som/sika --data.camera-type droid_recon --data.seq_name sika

python vis_depths_lh.py data:custom --data.data_dir /root/autodl-tmp/data/ac3d2som/sika --data.camera-type droid_recon --data.seq_name sika
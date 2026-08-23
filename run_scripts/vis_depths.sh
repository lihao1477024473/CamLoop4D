# 遇到mask None

cd ..

# iphone
# python vis_depths.py data:iphone --data.data_dir /root/autodl-tmp/data/paper-windmill


# custom:garden_flythrough_seva_16;cat
# python vis_depths.py data:custom --data.data_dir /root/autodl-tmp/data/seva2som --data.camera-type droid_recon --data.seq_name cat

# name="sika"
# python vis_depths.py data:custom --data.data_dir /root/autodl-tmp/data/ac3d2som/sika --data.camera-type droid_recon --data.seq_name sika


# name="garden_flythrough"
# dirData="seva2som/${name}"
# python vis_depths.py data:custom --data.data_dir /root/autodl-tmp/data/${dirData} --data.camera-type droid_recon --data.seq_name ${name}


# name="women"
# nameFolder="select"
# dataAgl="ac3d2som" # seva2som,ac3d2som
# dirData="${dataAgl}/${nameFolder}"
# python vis_depths.py data:custom --data.data_dir /root/autodl-tmp/data/${dirData} --data.camera-type droid_recon --data.seq_name ${name}

# retriever2
name="graden_zoom-out"
nameFolder="graden"
dataAgl="seva2som" # seva2som,ac3d2som
dirData="${dataAgl}/${nameFolder}"
python vis_depths.py data:custom --data.data_dir /root/autodl-tmp/data/${dirData} --data.camera-type droid_recon --data.seq_name ${name}
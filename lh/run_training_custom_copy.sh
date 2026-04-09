cd ..



dataType="ac3d2som" # ac3d2som
folder="maskAllOneCls_py" # maskAllOneCls,maskAllOneCls_py
dataName="sheep1" # cheetah
nameTrain="${dataName}_${folder}"
dataDir="${dataType}/${folder}"
python run_training.py \
  --work-dir /root/autodl-tmp/shape-of-motion/lh/${nameTrain} \
  --no-use_2dgs \
  data:custom \
  --data.data-dir /root/autodl-tmp/data/${dataDir} \
  --data.camera-type droid_recon \
  --data.seq_name ${dataName}

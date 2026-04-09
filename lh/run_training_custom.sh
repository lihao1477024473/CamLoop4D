cd ..


 # python run_training.py \
  # --work-dir /root/autodl-tmp/shape-of-motion/lh/custom \
  # --port 8080 \
  # data:custom \
  # --data.seq-name garden_flythrough_seva_16



# # To train custom dataset
# # garden_flythrough_seva_16;seq1;cat
#   dataName="sika"
# python run_training.py \
#   --work-dir /root/autodl-tmp/shape-of-motion/lh/${dataName} \
#   --no-use_2dgs \
#   data:custom \
#   --data.data-dir /root/autodl-tmp/data/ac3d2som/sika \
#   --data.camera-type droid_recon \
#   --data.seq_name ${dataName}



# # garden_flythrough;seq1;cat
# dataName="garden_flythrough"
# python run_training.py \
#   --work-dir /root/autodl-tmp/shape-of-motion/lh/${dataName} \
#   --use_2dgs \
#   data:custom \
#   --data.data-dir /root/autodl-tmp/data/seva2som/${dataName} \
#   --data.camera-type droid_recon \
#   --data.seq_name ${dataName}


#   # bear;cheetah,sika1,sika2,women
# dataName="women"
# dataDir="ac3d2som/select"
# python run_training.py \
#   --work-dir /root/autodl-tmp/shape-of-motion/lh/${dataName} \
#   --no-use_2dgs \
#   data:custom \
#   --data.data-dir /root/autodl-tmp/data/${dataDir} \
#   --data.camera-type droid_recon \
#   --data.seq_name ${dataName}


# lily-dragon_lemniscate,lily-dragon_move-backward,lily-dragon_move-right,lily-dragon_orbit,lily-dragon_spiral,lily-dragon_zoom-in,lily-dragon_zoom-out
# "blue-car_lemniscate,blue-car_spiral,blue-car_zoom-out
model="som-change"
dataType="ac3d2som" # ac3d2som;seva2som
folder="maskfg_select250625" # maskAllOneCls,maskAllOneCls_py,maskAll, maskAll_select250625;select;maskfg_select250625
dataName="tiger2_00018" # retriever1,lily-dragon_lemniscate,tiger2_00018,astronaut_00005
nameTrain="${dataName}_${folder}_any4dv350-b18"
dataDir="${dataType}/${folder}"
python run_training.py \
  --work-dir /root/autodl-tmp/${model}/lh/${nameTrain} \
  --no-use_2dgs \
  data:custom \
  --data.data-dir /root/autodl-tmp/data/${dataDir} \
  --data.camera-type droid_recon \
  --data.seq_name ${dataName}


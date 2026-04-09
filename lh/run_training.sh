cd ..

<<<<<<< Updated upstream
dataName="backpack" # paper-windmill,backpack;spin
out="lh39" # lh331；lh375
# python -m torch.distributed.run --nproc_per_node=2 run_training.py \
python run_training.py \
  --work-dir /root/autodl-tmp/som-change/${out}/${dataName}_b17 \
=======
dataName="paper-windmill" # paper-windmill,backpack;spin
model="any4d-v2.2"
out="lh22"
# python -m torch.distributed.run --nproc_per_node=2 run_training.py \
python run_training.py \
  --work-dir /root/autodl-tmp/${model}/${out}/${dataName} \
>>>>>>> Stashed changes
  --port 8080 \
  data:iphone \
  --data.data-dir /root/autodl-tmp/data/${dataName}/
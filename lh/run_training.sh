cd ..

dataName="backpack" # paper-windmill,backpack;spin
out="lh39" # lh331；lh375
# python -m torch.distributed.run --nproc_per_node=2 run_training.py \
python run_training.py \
  --work-dir /root/autodl-tmp/som-change/${out}/${dataName}_b17 \
  --port 8080 \
  data:iphone \
  --data.data-dir /root/autodl-tmp/data/${dataName}/
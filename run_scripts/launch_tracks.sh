cd ..
cd preproc

data_root='/root/autodl-tmp/data/seva2som'

python launch_tracks.py \
    --img-dirs ${data_root}/images/** --gpus 0
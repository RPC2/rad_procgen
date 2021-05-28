env_name=coinrun

CUDA_VISIBLE_DEVICES=0 python -m train_procgen.train \
  --env_name ${env_name} \
  --num_levels 100 \
  --exp_name demo \
  --load_path /home/cynthiachen/from-github/rad_procgen/log/${env_name}/nlev_100_mode_easy/normal/try1/checkpoints/01488

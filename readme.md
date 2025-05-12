train:
python scripts/segmentation_train.py --data_name mydata --data_dir data/mydata --out_dir /home/nas2/biod/piankexin/FetalSeg/result/0327 --image_size 256 --num_channels 64 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 8 

sample:
CUDA_VISIBLE_DEVICES=4 python scripts/segmentation_sample.py --data_name mydata --data_dir data/mydata  --out_dir output/mydata/sampling0314 --model_path /home/nas2/biod/piankexin/My_model/results/results0314/emasavedmodel_0.9999_000800.pt --image_size 256 --num_channels 64 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5

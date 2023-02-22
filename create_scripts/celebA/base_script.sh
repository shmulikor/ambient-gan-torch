HPARAMS="
dataset=celebA,\
measurement_type=drop_independent,\
drop_prob=0.0,\
patch_size=32,\
blur_radius=1.0,\
blur_filter_size=5,\
additive_noise_std=0.0,\
num_angles=1,\
model_type=dcgan,\
z_dim=100,\
gp_lambda=10.0,\
batch_size=64,\
g_lr=0.0002,\
d_lr=0.0002,\
opt_param1=0.5,\
opt_param2=0.999,\
results_dir=./results/,\
epochs=100,\
"

python src/main.py \
    --hparams $HPARAMS

HPARAMS="
dataset=mnist,\
measurement_type=drop_independent,\
drop_prob=0.0,\
patch_size=14,\
blur_radius=1.0,\
blur_filter_size=5,\
additive_noise_std=0.0,\
num_angles=1,\
model_type=wgangp,\
z_dim=100,\
gp_lambda=10.0,\
batch_size=64,\
g_lr=0.0001,\
d_lr=0.0001,\
opt_param1=0.5,\
opt_param2=0.999,\
results_dir=./results/,\
epochs=50,\
"

python src/main.py \
    --hparams $HPARAMS

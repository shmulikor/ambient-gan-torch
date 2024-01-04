mkdir scripts


# -----------

# mnist wgangp drop independent
python ./create_scripts/create_scripts.py \
   --base-script-path ./create_scripts/mnist/base_script.sh \
   --scripts-base-dir ./scripts/ \
   --grid-path ./create_scripts/mnist/grid_wgangp_drop_independent.sh

# mnist wgangp blur + noise
python ./create_scripts/create_scripts.py \
   --base-script-path ./create_scripts/mnist/base_script.sh \
   --scripts-base-dir ./scripts/ \
   --grid-path ./create_scripts/mnist/grid_wgangp_blur_addnoise.sh

# mnist wgangp keep patch
python ./create_scripts/create_scripts.py \
   --base-script-path ./create_scripts/mnist/base_script.sh \
   --scripts-base-dir ./scripts/ \
   --grid-path ./create_scripts/mnist/grid_wgangp_keep_patch.sh

# mnist wgangp extract patch
python ./create_scripts/create_scripts.py \
   --base-script-path ./create_scripts/mnist/base_script.sh \
   --scripts-base-dir ./scripts/ \
   --grid-path ./create_scripts/mnist/grid_wgangp_extract_patch.sh

# mnist wgangp drop patch
python ./create_scripts/create_scripts.py \
   --base-script-path ./create_scripts/mnist/base_script.sh \
   --scripts-base-dir ./scripts/ \
   --grid-path ./create_scripts/mnist/grid_wgangp_drop_patch.sh


# -----------

# # celebA dcgan drop_independent
# python ./create_scripts/create_scripts.py \
#     --base-script-path ./create_scripts/celebA/base_script.sh \
#     --scripts-base-dir ./scripts/ \
#     --grid-path ./create_scripts/celebA/grid_dcgan_drop_independent.sh

# # celebA dcgan blur + noise
# python ./create_scripts/create_scripts.py \
#     --base-script-path ./create_scripts/celebA/base_script.sh \
#     --scripts-base-dir ./scripts/ \
#     --grid-path ./create_scripts/celebA/grid_dcgan_blur_addnoise.sh

# # celebA dcgan patches
# python ./create_scripts/create_scripts.py \
#     --base-script-path ./create_scripts/celebA/base_script.sh \
#     --scripts-base-dir ./scripts/ \
#     --grid-path ./create_scripts/celebA/grid_dcgan_patches.sh


# -----------

# QSM_cosmos wgangp drop_independent
#python ./create_scripts/create_scripts.py \
#    --base-script-path ./create_scripts/QSM_cosmos/base_script.sh \
#    --scripts-base-dir ./scripts/ \
#    --grid-path ./create_scripts/QSM_cosmos/grid_wgangp_drop_independent.sh


# Make sure everything in scripts is executable
chmod +x ./scripts/*.sh

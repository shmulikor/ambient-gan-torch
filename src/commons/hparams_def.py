# from hparams import HParams
import re

def get_hparams(args=None):

    # Create a HParams object specifying the names and values of the model hyperparameters:

    # hparams = tf.contrib.training.HParams(
    hparams = {
        # task
        'dataset': 'mnist',
        'measurement_type': 'drop_independent',
        'drop_prob': 0.5,  # drop probability
        'patch_size': 10,  # size of patch to drop
        'blur_radius': 1.0,  # Radius for gaussian blurring
        'blur_filter_size': 3,  # Size of the blurring filter
        'additive_noise_std': 0.5,  # std deviation of noise to add
        'num_angles': 1,  # Number of rotate + project measurements

        # model
        'model_type': 'wgangp',
        'z_dim': 100,
        'gp_lambda': 10.0,  # gradient penalty lambda

        # optimization
        'batch_size': 64,  # how many examples are processed together
        'g_lr': 0.0002,  # Learning rate for the generator
        'd_lr': 0.0002,  # Learning rate for the disciminator
        'opt_param1': 0.5,  # parameter1 to optimizer
        'opt_param2': 0.999,  # parameter2 to optimizer

        # monitoring, saving, running
        'results_dir': './results/', # Where to store the results
        'use_one_ckpt': True,
        'use_saved_model': True,

        'epochs': 50,

    }

    # Override hyperparameters values by parsing the command line

    if args.hparams is not None:
        re_float = re.compile(r"(^\d+\.\d+$|^\.\d+$)")
        re_int = re.compile(r"(^[1-9]+\d*$|^0$)")
        new_args = args.hparams.split(',')
        for new_arg in new_args[:-1]:
            k, v = new_arg.split('=')
            print(k, v)
            v = float(v) if re.match(re_float, v) is not None else int(v) if re.match(re_int, v) else v
            hparams[k] = v
        print(hparams)
        print('-------------------------------')
    return hparams

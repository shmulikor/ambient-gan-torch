# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914

"""Functions to compute expt_dir from hparams"""

def get_task_dir(hparams):

    if hparams['measurement_type'] in ['drop_independent', 'drop_row', 'drop_col', 'drop_rowcol']:
        task_dir = '{}_p{}/'.format(
            hparams['measurement_type'],
            hparams['drop_prob'],
        )
    elif hparams['measurement_type'] in ['drop_patch', 'keep_patch', 'extract_patch']:
        task_dir = '{}_k{}/'.format(
            hparams['measurement_type'],
            hparams['patch_size'],
        )
    elif hparams['measurement_type'] in ['blur_addnoise']:
        task_dir = '{}_br{}_bfs{}_anstd{}/'.format(
            hparams['measurement_type'],
            hparams['blur_radius'],
            hparams['blur_filter_size'],
            hparams['additive_noise_std'],
        )
    elif hparams['measurement_type'] in ['pad_rotate_project', 'pad_rotate_project_with_theta']:
        task_dir = '{}_na{}/'.format(
            hparams['measurement_type'],
            hparams['num_angles'],
        )
    else:
        raise NotImplementedError

    return task_dir


def get_model_dir(hparams):

    if hparams['model_type'] == 'dcgan':
        model_dir = '{}_zd{}/'.format(
            hparams['model_type'],
            hparams['z_dim'],
        )
    elif hparams['model_type'] == 'wgangp':
        model_dir = '{}_zd{}_gpl{}/'.format(
            hparams['model_type'],
            hparams['z_dim'],
            hparams['gp_lambda'],
        )
    else:
        raise NotImplementedError

    return model_dir


def get_opt_dir(hparams):
    opt_dir = 'bs{}_glr{}_dlr{}_p{}_p{}/'.format(
        hparams['batch_size'],
        hparams['g_lr'],
        hparams['d_lr'],
        hparams['opt_param1'],
        hparams['opt_param2'],
    )
    return opt_dir


def get_expt_dir(hparams):
    dataset_dir = '{}/'.format(hparams['dataset'])
    task_dir = get_task_dir(hparams)
    model_dir = get_model_dir(hparams)
    opt_dir = get_opt_dir(hparams)
    expt_dir = dataset_dir + task_dir + model_dir + opt_dir
    return expt_dir

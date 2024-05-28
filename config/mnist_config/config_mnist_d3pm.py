import ml_collections
import os


# config_bert_001: Param: 7 226 883
def get_config():
    save_directory = "SavedModels/MNIST/"

    config = ml_collections.ConfigDict()
    config.save_location = save_directory
    config.device = "cpu"
    config.distributed = False

    config.training = training = ml_collections.ConfigDict()
    training.train_step_name = "Standard"
    training.n_iters = 600000  # 2000 #2000000
    training.clip_grad = True
    training.grad_norm = 1
    training.warmup = 0  # 5000

    # data config
    config.data = data = ml_collections.ConfigDict()
    data.name = "DiscreteMNIST"
    data.train = True
    data.download = True
    data.S = 256
    data.batch_size = 64
    data.shuffle = True
    data.image_size = 28
    data.shape = [1, data.image_size, data.image_size]
    data.random_flips = True
    data.use_augm = False

    # model config
    config.model = model = ml_collections.ConfigDict()
    model.name = "GaussianTargetRateImageX0PredEMAPaul"
    model.padding = False
    model.ema_decay = 0.9999  # 0.9999

    model.ch = 32
    model.num_res_blocks = 2
    model.ch_mult = [1, 2]  # [1, 2, 2, 2]
    model.input_channels = 1
    model.scale_count_to_put_attn = 1
    model.data_min_max = [0, 255]
    model.dropout = 0.1
    model.skip_rescale = True
    model.time_embed_dim = model.ch
    model.time_scale_factor = 1000
    model.fix_logistic = False
    model.model_output = "logits"  # logistic_pars'
    model.num_heads = 1
    model.attn_resolutions = [int(model.ch / 2)]
    model.concat_dim = data.image_size * data.image_size * 1
    model.padding = False
    model.is_img = True

    # diffusion betas
    model.type = (
        "linear"  # start, stop only relevant for linear, power, jsdtrunc schedules.
    )
    model.start = 1e-4  # 1e-4 gauss, 0.02 uniform
    model.stop = 0.02  # 0.02, gauss, 1. uniform
    model.num_timesteps = 1000

    # Settings used in diffusion_categorical.py
    model.model_prediction = "x_start"  # 'x_start','xprev'
    # 'gaussian','uniform','absorbing'
    model.transition_mat_type = "gaussian"
    model.transition_bands = None
    model.loss_type = "hybrid"  # kl,cross_entropy_x_start, hybrid
    model.hybrid_coeff = 0.001
    model.model_output = "logits"
    model.num_pixel_vals = 256
    model.device = "cpu"

    config.optimizer = optimizer = ml_collections.ConfigDict()
    optimizer.name = "Adam"
    optimizer.lr = 2e-4  # 2e-4

    config.saving = saving = ml_collections.ConfigDict()
    saving.sample_plot_path = os.path.join(save_directory, "PNGs")
    saving.checkpoint_freq = 5000

    return config

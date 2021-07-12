import argparse

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--h5ad_path',
        type=str,
        default="filtered_azimuth_and_geninus.h5ad",
        help='path to a h5ad data file.'
    )
    parser.add_argument(
        '--out_path',
        type=str,
        required=True,
        help='path to a h5ad data file.'
    )
    parser.add_argument(
        '--trained_model',
        type=str,
        default=None,
        required=False,
        help='path to a directory of pretrained model.'
    )
    parser.add_argument(
        '--groupby',
        type=str,
        default="celltype.l2",
        help='dictionary key for label'
    )
    parser.add_argument(
        '--domain_groupby',
        type=str,
        default="domain",
        help='dictionary key for label'
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '--optimizer_name',
        type=str,
        default="adadelta",
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        '--mixup_alpha',
        type=float,
        default=0.3,
    )
    parser.add_argument(
        '--unsup_max_weight',
        type=float,
        default=0.3,
    )
    parser.add_argument(
        '--unsup_mean_teacher',
        type=lambda x: (str(x).lower() == 'true'), 
        default=False, 
    )
    parser.add_argument(
        '--class_balance_solution',
        type=str,
        default='',
        help="weighted or balanced"
    )
    # parser.add_argument(
    #     '--weighted_classes', 
    #     type=lambda x: (str(x).lower() == 'true'), 
    #     default=False, 
    # )
    # parser.add_argument(
    #     '--balanced_classes', 
    #     type=lambda x: (str(x).lower() == 'true'), 
    #     default=False, 
    # )
    parser.add_argument(
        '--ssl_method',
        type=str,
        default="mixmatch",
    )
    parser.add_argument(
        '--ssl_kwargs/augment_pseudolabels',
        type=lambda x: (str(x).lower() == 'true'), 
        default=False, 
    )
    parser.add_argument(
        '--ssl_kwargs/augment',
        type=str, 
        default="log1p_drop", 
    )
    parser.add_argument(
        '--ssl_kwargs/unsup_criterion',
        type=str, 
        default="mse", 
    )
    parser.add_argument(
        '--ssl_kwargs/n_augmentations',
        type=int, 
        default=1, 
    )
    parser.add_argument(
        '--ssl_kwargs/T',
        type=float, 
        default=0.5, 
    )
    parser.add_argument(
        '--ssl_kwargs/ramp_epochs',
        type=int, 
        default=100, 
    )
    parser.add_argument(
        '--ssl_kwargs/burn_in_epochs',
        type=int, 
        default=0, 
    )
    parser.add_argument(
        '--ssl_kwargs/dan_criterion',
        type=lambda x: (str(x).lower() == 'true'), 
        default=True, 
    )
    parser.add_argument(
        '--ssl_kwargs/dan_ramp_epochs',
        type=int, 
        default=20, 
    )
    parser.add_argument(
        '--ssl_kwargs/dan_max_weight',
        type=float, 
        default=0.1, 
    )
    parser.add_argument(
        '--ssl_kwargs/min_epochs',
        type=int, 
        default=20, 
    )    
    parser.add_argument(
        '--ssl_kwargs/pseudolabel_min_confidence',
        type=float, 
        default=0.5, 
    )
    parser.add_argument(
        '--model_kwargs/n_hidden',
        type=int, 
        default=256, 
    )
    parser.add_argument(
        '--model_kwargs/n_layers',
        type=float, 
        default=2, 
    )
    parser.add_argument(
        '--model_kwargs/init_dropout',
        type=float, 
        default=0.0, 
    )
    parser.add_argument(
        '--model_kwargs/residual', 
        type=lambda x: (str(x).lower() == 'true'), 
        default=False, 
    )
    parser.add_argument(
        '--in_metalearner', 
        type=lambda x: (str(x).lower() == 'true'), 
        default=False, 
        help='whether work in meta learner'
    )

    parser.add_argument(
        '--verbose',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--tensorboard', 
        type=lambda x: (str(x).lower() == 'true'), 
        default=True, 
        help='make tensorboard log'
    )

    
    return parser
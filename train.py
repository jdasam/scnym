import numpy as np
import scipy
import pandas as pd
import anndata
import scanpy as sc
import scnym.utils as utils
import scnym.model as model
import os
import os.path as osp
import _pickle as pickle
import random
from scnym import main
from anndata import AnnData
from metalearner_parser import make_parser
UNLABELED_TOKEN = 'Unlabeled'


def scnym_train(
    adata: AnnData,
    config: dict,
) -> None:
    '''Train an scNym model. 
    
    Parameters
    ----------
    adata : AnnData
        [Cells, Genes] experiment containing annotated
        cells to train on.
    config : dict
        configuration options.
    
    Returns
    -------
    None.
    Saves model outputs to `config["out_path"]` and adds model results
    to `adata.uns["scnym_train_results"]`.
    
    Notes
    -----
    This method should only be directly called by advanced users.
    Most users should use `scnym_api`.
    
    See Also
    --------
    scnym_api
    '''
    # determine if unlabeled examples are present
    n_unlabeled = np.sum(
        adata.obs[config['groupby']] == UNLABELED_TOKEN
    )
    if n_unlabeled == 0:
        print('No unlabeled data was found.')
        print(f'Did you forget to set some examples as `"{UNLABELED_TOKEN}"`?')
        print('Proceeding with purely supervised training.')
        print()
        
        unlabeled_counts = None
        unlabeled_genes  = None
        
        X = utils.get_adata_asarray(adata)
        y = pd.Categorical(
            np.array(adata.obs[config['groupby']]),
            categories=np.unique(adata.obs[config['groupby']]),
        ).codes
        class_names = np.unique(adata.obs[config['groupby']])
        # set all samples for training
        train_adata = adata
    else:
        print(f'{n_unlabeled} unlabeled observations found.')
        print('Using unlabeled data as a target set for semi-supervised, adversarial training.')
        print()
        
        target_bidx = adata.obs[config['groupby']] == UNLABELED_TOKEN
        
        train_adata = adata[~target_bidx, :]
        target_adata = adata[target_bidx, :]
        
        print('training examples: ', train_adata.shape)
        print('target   examples: ', target_adata.shape)
        
        X = utils.get_adata_asarray(train_adata)
        y = pd.Categorical(
            np.array(train_adata.obs[config['groupby']]),
            categories=np.unique(train_adata.obs[config['groupby']]),
        ).codes
        unlabeled_counts = utils.get_adata_asarray(target_adata)
        class_names = np.unique(train_adata.obs[config['groupby']])
        
    print('X: ', X.shape)
    print('y: ', y.shape)
    
    if 'scNym_split' not in adata.obs_keys():
        # perform a 90/10 train test split
        traintest_idx = np.random.choice(
            X.shape[0],
            size=int(np.floor(0.9*X.shape[0])),
            replace=False
        )
        val_idx = np.setdiff1d(np.arange(X.shape[0]), traintest_idx)
    else:
        train_idx = np.where(
            train_adata.obs['scNym_split'] == 'train'
        )[0]
        test_idx = np.where(
            train_adata.obs['scNym_split'] == 'test',
        )[0]
        val_idx = np.where(
            train_adata.obs['scNym_split'] == 'val'
        )[0]
        
        if len(train_idx) < 100 or len(test_idx) < 10 or len(val_idx) < 10:
            msg = 'Few samples in user provided data split.\n'
            msg += f'{len(train_idx)} training samples.\n'
            msg += f'{len(test_idx)} testing samples.\n'
            msg += f'{len(val_idx)} validation samples.\n'
            msg += 'Halting.'
            raise RuntimeError(msg)
        # `fit_model()` takes a tuple of `traintest_idx`
        # as a training index and testing index pair.
        traintest_idx = (
            train_idx,
            test_idx,
        )
        
    # check if domain labels were manually specified
    if config.get('domain_groupby', None) is not None:
        domain_groupby = config['domain_groupby']
        # check that the column actually exists
        if domain_groupby not in adata.obs.columns:
            msg = f'no column {domain_groupby} exists in `adata.obs`.\n'
            msg += 'if domain labels are specified, a matching column must exist.'
            raise ValueError(msg)
        # get the label indices as unique integers using pd.Categorical
        # to code each unique label with an int
        domains = np.array(
            pd.Categorical(
                adata.obs[domain_groupby],
                categories=np.unique(adata.obs[domain_groupby]),
            ).codes,
            dtype=np.int32,
        )
        # split domain labels into source and target sets for `fit_model`
        input_domain = domains[~target_bidx]
        unlabeled_domain = domains[target_bidx]
        print('Using user provided domain labels.')
        n_source_doms = len(np.unique(input_domain))
        n_target_doms = len(np.unique(unlabeled_domain))
        print(
            f'Found {n_source_doms} source domains and {n_target_doms} target domains.'
        )
    else:
        # no domains manually supplied, providing `None` to `fit_model`
        # will treat source data as one domain and target data as another
        input_domain = None
        unlabeled_domain = None
        
    # check if pre-trained weights should be used to initialize the model
    if config['trained_model'] is None:
        pretrained = None
    elif 'pretrained_' in config['trained_model']:
        msg = 'pretrained model fetching is not supported for training.'
        raise NotImplementedError(msg)
    else:
        # setup a prediction model
        pretrained = osp.join(
            config['trained_model'],
            '00_best_model_weights.pkl',
        )
        if not osp.exists(pretrained):
            msg = f'{pretrained} file not found.'
            raise FileNotFoundError(msg)
        
    acc, loss = main.fit_model(
        X=X,
        y=y,
        traintest_idx=traintest_idx,
        val_idx=val_idx,
        batch_size=config['batch_size'],
        n_epochs=config['n_epochs'],
        lr=config['lr'],
        optimizer_name=config['optimizer_name'],
        weight_decay=config['weight_decay'],
        ModelClass=model.CellTypeCLF,
        out_path=config['out_path'],
        mixup_alpha=config['mixup_alpha'],
        unlabeled_counts=unlabeled_counts,
        input_domain=input_domain,
        unlabeled_domain=unlabeled_domain,
        unsup_max_weight=config['unsup_max_weight'],
        unsup_mean_teacher=config['unsup_mean_teacher'],
        ssl_method=config['ssl_method'],
        ssl_kwargs=config['ssl_kwargs'],
        pretrained=pretrained,
        patience=config.get('patience', None),
        save_freq=config.get('save_freq', None),
        tensorboard=config.get('tensorboard', False),
        weighted_classes=config['weighted_classes'],
        balanced_classes=config['balanced_classes'],
        verbose=config['verbose'],
        in_metalearner=config['in_metalearner'],
        **config['model_kwargs'],
    )
    
    # add the final model results to `adata`
    results = {
        'model_path': osp.realpath(osp.join(config['out_path'], '00_best_model_weights.pkl')),
        'final_acc': acc,
        'final_loss': loss,
        'n_genes': adata.shape[1],
        'n_cell_types': len(np.unique(y)),
        'class_names': class_names,
        'gene_names': adata.var_names.tolist(),
        'model_kwargs': config['model_kwargs'],
        'traintest_idx': traintest_idx,
        'val_idx': val_idx,
    }
    assert osp.exists(results['model_path'])
    
    adata.uns['scNym_train_results'] = results
    
    # save the final model results to disk
    train_results_path = osp.join(
        config['out_path'], 
        'scnym_train_results.pkl',
    )

    with open(train_results_path, 'wb') as f:
        pickle.dump(
            results,
            f
        )
    return


if __name__ == "__main__":
    parser = make_parser()
    configs = parser.parse_args()

    # path = "/home/svcapp/tbrain_x/azimuth/pbmc_multimodal_raw.h5ad"
    # unlabeled_path = "/home/svcapp/tbrain_x/SKT_data_corrected/IO_response/geninus_raw_final.h5ad"
    # path = "filtered_azimuth_and_geninus.h5ad"

#     configs = {
#     'out_path': "ssl_balanced/",
#     'groupby': "celltype.l2",
#     'n_epochs': 1000,
#     'patience': 40,
#     'lr': 0.1,
#     'optimizer_name': 'adadelta',
#     'weight_decay': 1e-4,
#     'batch_size': 512,
#     'mixup_alpha': 0.3,
#     'unsup_max_weight': 0.5,
#     'unsup_mean_teacher': False,
#     'weighted_classes': False,
#     'balanced_classes': True,
#     'verbose': 0,
#     'ssl_method': 'mixmatch',
#     'ssl_kwargs': {
#         'augment_pseudolabels': False,
#         'augment': 'log1p_drop',
#         'unsup_criterion': 'mse',
#         'n_augmentations': 1,
#         'T': 0.5,
#         'ramp_epochs': 100,
#         'burn_in_epochs': 0,
#         'dan_criterion': True,
#         'dan_ramp_epochs': 20,
#         'dan_max_weight': 0.1,
#         'min_epochs': 20,
#         'pseudolabel_min_confidence':0.5
#     },
#     'model_kwargs' : {
#         'n_hidden': 256,
#         'n_layers': 2,
#         'init_dropout': 0.0,
#         'residual': False,
#     },
#     'tensorboard': True,
# }

    # unlabeled_adata = anndata.read_h5ad(unlabeled_path)
    # unlabeled_adata.X = scipy.sparse.csr_matrix(unlabeled_adata.X, dtype=np.float32)
    # sc.pp.normalize_per_cell(unlabeled_adata, counts_per_cell_after=2e3)

    adata = anndata.read_h5ad(configs.h5ad_path)
    # adata = anndata.read_h5ad(path)
    # sc.pp.normalize_per_cell(adata, counts_per_cell_after=2e3)
    # sc.pp.filter_genes(adata, min_cells=20000)
    # sc.pp.filter_cells(adata, min_genes=100)
    
    # num_filtered = 161753

    # sample_idx = random.sample(range(num_filtered, adata.shape[0]), (adata.shape[0]-num_filtered)//4)
    # adata = anndata.concat( [adata[:num_filtered], adata[sample_idx]])
    # adata.obs["celltype.l2"][sample_idx] = "Unlabeled"

    # adata.obs["domain"] = "azimuth"
    # gene_names = adata.var_names.tolist()
    # gene_idx = np.zeros(len(gene_names), dtype=np.int32)
    # for i, gene in enumerate(gene_names):
    #     gene_idx[i] = np.where(unlabeled_adata.var.index == gene_names[i])[0]
    # unlabeled_adata = unlabeled_adata[:, gene_idx]
    # unlabeled_adata.obs["celltype.l2"] = "Unlabeled"
    # unlabeled_adata.obs["domain"] = "geninus"

    # sc.pp.filter_cells(unlabeled_adata, min_genes=100)

    # num_unlabel = unlabeled_adata.shape[0]
    # end_idx = num_unlabel // 5

    # for i in range(4):
    #     print(i)
    #     adata = anndata.concat([adata, unlabeled_adata[:end_idx]])
    #     print(f"{i}th concat finished")
    #     unlabeled_adata = unlabeled_adata[end_idx:]
    # print("try concat")
    # adata = anndata.concat([adata, unlabeled_adata])
    # print("concat end")
    # del unlabeled_adata
    # print("del end")

    # configs['trained_model'] = None
    # configs['domain_groupby'] = None
    # configs['trained_model'] = "sl_only/"
    # configs['domain_groupby'] = "domain"
    os.makedirs(configs.out_path, exist_ok=True)

    print("train start")
    config_dict = vars(configs)
    config_dict['ssl_kwargs'] = {}
    config_dict['model_kwargs'] = {}
    keys = list(config_dict.keys())
    for key in keys:
        if 'ssl_kwargs/' in key:
            config_dict['ssl_kwargs'][key.split('/')[1]] = config_dict.pop(key)
        if 'model_kwargs/' in key:
            config_dict['model_kwargs'][key.split('/')[1]] = config_dict.pop(key)
    class_balance_solution = config_dict.pop('class_balance_solution')
    config_dict['balanced_classes'] = class_balance_solution == "balanced"
    config_dict['weighted_classes'] = class_balance_solution == "weighted"
        
    scnym_train(adata, config_dict)
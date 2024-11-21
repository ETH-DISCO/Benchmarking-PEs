from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('posenc')
def set_cfg_posenc(cfg):
    """Extend configuration with positional encoding options.
    """

    # Argument group for each Positional Encoding class.
    cfg.posenc_LapPE = CN()
    cfg.posenc_SignNet = CN()
    cfg.posenc_RWSE = CN()
    cfg.posenc_HKdiagSE = CN()
    cfg.posenc_ElstaticSE = CN()
    cfg.posenc_EquivStableLapPE = CN()
    cfg.posenc_GPSE = CN()
    cfg.posenc_RRWP = CN()
    cfg.posenc_SVD = CN()
    cfg.posenc_PPR = CN()
    cfg.posenc_WLPE = CN()
    cfg.posenc_GCKN = CN()
    cfg.posenc_RWDIFF = CN()
    cfg.posenc_GD = CN()

    # Common arguments to all PE types.
    for name in ['posenc_LapPE', 'posenc_SignNet',
                 'posenc_RWSE', 'posenc_HKdiagSE', 'posenc_ElstaticSE',
                 'posenc_RRWP', 'posenc_SVD', 'posenc_PPR',
                 'posenc_WLPE', 'posenc_GCKN', 'posenc_GPSE',
                 'posenc_RWDIFF', 'posenc_GD',
                 ]:
        pecfg = getattr(cfg, name)
        # Use extended positional encodings
        pecfg.enable = False
        pecfg.add_self_loops = False

        # Neural-net model type within the PE encoder:
        # 'DeepSet', 'Transformer', 'Linear', 'none', ...
        pecfg.model = 'none'

        # Size of Positional Encoding embedding
        pecfg.dim_pe = 16

        # Number of layers in PE encoder model
        pecfg.layers = 3

        # Number of attention heads in PE encoder when model == 'Transformer'
        pecfg.n_heads = 4

        # Number of layers to apply in LapPE encoder post its pooling stage
        pecfg.post_layers = 0

        # Choice of normalization applied to raw PE stats: 'none', 'BatchNorm'
        pecfg.raw_norm_type = 'none'

        # In addition to appending PE to the node features, pass them also as
        # a separate variable in the PyG graph batch object.
        pecfg.pass_as_var = False


    if name == "posenc_RWDIFF":
        pecfg = getattr(cfg, name)

        pecfg.pos_enc_dim = 10

        pecfg.dim_pe = 10
    # if name == 'posenc_NODE2VEC':
    #
    #     pecfg = getattr(cfg, name)
    #
    #     pecfg.node2vec.dimensions = 64
    #
    #     pecfg.node2vec.walk_length = 30
    #
    #     pecfg.node2vec.num_walks = 200
    #
    #     pecfg.node2vec.workers=4
    #
    #     pecfg.node2vec.window = 10
    #
    #     pecfg.node2vec.min_count = 1
    #
    #     pecfg.node2vec.batch_words = 4

    if name == 'posenc_WLPE':
        pecfg = getattr(cfg, name)

        pecfg.dh = 100

    if name == 'posenc_GCKN':
        pecfg = getattr(cfg, name)

        pecfg.normalization = True

        pecfg.beta = 1.

        pecfg.p = 3

        pecfg.use_edge_attr = False

        pecfg.method = "diffusion"

    # Config for pretrained GNN P/SE encoder
    cfg.posenc_GPSE.model_dir = None
    cfg.posenc_GPSE.accelerator = "default"
    cfg.posenc_GPSE.rand_type = 'NormalSE'
    cfg.posenc_GPSE.use_repr = False  # use one layer before the output if True
    # What representation to use. 'one_layer_before' uses the representation of
    # the second to last layer in the poas_mp module as the repr. 'no_post_mp'
    # uses the input representation to the post_mp module as the repr, in other
    # words, no_post_mp skips the last pos_mp module.
    cfg.posenc_GPSE.repr_type = "one_layer_before"
    cfg.posenc_GPSE.virtual_node = False
    cfg.posenc_GPSE.input_dropout_be = 0.0
    cfg.posenc_GPSE.input_dropout_ae = 0.0
    cfg.posenc_GPSE.save = False
    cfg.posenc_GPSE.from_saved = False
    cfg.posenc_GPSE.tag = "1.0"

    # Loader for each graph
    cfg.posenc_GPSE.loader = CN()
    cfg.posenc_GPSE.loader.type = "full"
    cfg.posenc_GPSE.loader.num_neighbors = [30, 20, 10]
    cfg.posenc_GPSE.loader.fill_num_neighbors = 5
    cfg.posenc_GPSE.loader.batch_size = 1024

    # Multi MLP head hidden dimension. If None, set as the same as gnn.dim_inner
    cfg.gnn.multi_head_dim_inner = None
    cfg.posenc_GPSE.gnn_cfg = CN(cfg.gnn.copy())

    # cfg.posenc_SVD.calculated_dim = 8
    # Config for EquivStable LapPE
    cfg.posenc_EquivStableLapPE.enable = False
    cfg.posenc_EquivStableLapPE.raw_norm_type = 'none'

    # Config for Laplacian Eigen-decomposition for PEs that use it.
    for name in ['posenc_LapPE', 'posenc_SignNet', 'posenc_EquivStableLapPE',
            'posenc_SVD', 'posenc_PPR', 'posenc_WLPE',
            'posenc_GCKN']:
        pecfg = getattr(cfg, name)
        pecfg.eigen = CN()

        # The normalization scheme for the graph Laplacian: 'none', 'sym', or 'rw'
        pecfg.eigen.laplacian_norm = 'sym'

        # The normalization scheme for the eigen vectors of the Laplacian
        pecfg.eigen.eigvec_norm = 'L2'

        # Maximum number of top smallest frequencies & eigenvectors to use
        pecfg.eigen.max_freqs = 10

    # Config for SignNet-specific options.
    cfg.posenc_SignNet.phi_out_dim = 4
    cfg.posenc_SignNet.phi_hidden_dim = 64

    for name in ['posenc_RWSE', 'posenc_HKdiagSE', 'posenc_ElstaticSE']:
        pecfg = getattr(cfg, name)

        # Config for Kernel-based PE specific options.
        pecfg.kernel = CN()

        # List of times to compute the heat kernel for (the time is equivalent to
        # the variance of the kernel) / the number of steps for random walk kernel
        # Can be overridden by `posenc.kernel.times_func`
        pecfg.kernel.times = []

        # Python snippet to generate `posenc.kernel.times`, e.g. 'range(1, 17)'
        # If set, it will be executed via `eval()` and override posenc.kernel.times
        pecfg.kernel.times_func = ''

    # Override default, electrostatic kernel has fixed set of 10 measures.
    cfg.posenc_ElstaticSE.kernel.times_func = 'range(10)'

    # ----------------- Note: RRWP --------------
    cfg.posenc_RRWP.enable = False
    cfg.posenc_RRWP.ksteps = 21
    cfg.posenc_RRWP.add_identity = True
    cfg.posenc_RRWP.spd = False




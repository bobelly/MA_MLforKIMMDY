import tensorflow.keras as ks
import pprint
import tensorflow as tf
import numpy as np
from kgcnn.layers.casting import ChangeTensorType, ChangeIndexing
from kgcnn.layers.conv.schnet_conv import SchNetInteraction
from kgcnn.layers.keras import Dense, Add, Concatenate, Dropout
from kgcnn.layers.mlp import MLP
from kgcnn.layers.gather import GatherNodes, GatherState, GatherNodesIngoing
from kgcnn.layers.pool.pooling import PoolingNodes
from kgcnn.layers.geom import NodeDistance, BesselBasisLayer, EdgeDirectionNormalized  
from kgcnn.layers.conv.painn_conv import PAiNNconv, PAiNNUpdate
from kgcnn.layers.conv.megnet_conv import MEGnetBlock
from kgcnn.utils.loss import ScaledMeanAbsoluteError
from utils import generate_standard_graph_input, generate_mol_graph_input, update_model_args, generate_node_embedding
from utils import CosineLearningRateScheduler as lr

# PaiNN model adapted for barrier energy from structure calculation
def painn_adapted(**kwargs):
    """Get PAiNN model.
    Args:
        **kwargs
    Returns:
        tf.keras.models.Model: PAiNN keras model.
    """
    model_args = kwargs
    model_default = {'input_node_shape': None, 'input_equiv_shape': None,
                     'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 128}},
                     'output_embedding': {"output_mode": 'graph', "output_tensor_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True], "units": [128, 1],
                                    "activation": ['swish', 'linear']},
                     'bessel_basis': {'num_radial': 20, 'cutoff': 5.0, 'envelope_exponent': 5},
                     'pooling_args': {'pooling_method': 'sum'},
                     'depth': 3,
                     'verbose': 1
                     }
    m = update_model_args(model_default, model_args)
    if m['verbose'] > 0:
        print("INFO: Updated functional make model kwargs:")
        pprint.pprint(m)

    # local variables
    input_node_shape = m['input_node_shape']
    input_equiv_shape = m['input_equiv_shape']
    input_embedding = m['input_embedding']
    bessel_basis = m['bessel_basis']
    depth = m['depth']
    output_embedding = m['output_embedding']
    pooling_args = m['pooling_args']
    output_mlp = m['output_mlp']

    # Make input
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    equiv_input = ks.layers.Input(shape=input_equiv_shape, name='equiv_input', dtype="float32", ragged=True)
    xyz_input = ks.layers.Input(shape=[None, 3], name='xyz_input', dtype="float32", ragged=True)
    bond_index_input = ks.layers.Input(shape=[None, 2], name='bond_index_input', dtype="int64", ragged=True)
    node_radical_index = ks.layers.Input(shape=(None, 2), name='node_radical_input', dtype="int64", ragged=True)
    eri_n = node_radical_index
    edge_radical_index = ks.layers.Input(shape=(None, 1), name='edge_radical_input', dtype="int64", ragged=True)
    eri_e = edge_radical_index

    # Embedding
    z = generate_node_embedding(node_input, input_node_shape, input_embedding['nodes'])
    edi = bond_index_input
    x = xyz_input
    v = equiv_input

    rij = EdgeDirectionNormalized()([x, edi])
    d = NodeDistance()([x, edi])
    rbf = BesselBasisLayer(**bessel_basis)(d)

    for i in range(depth):
        # Message
        ds, dv = PAiNNconv(units=128)([z, v, rbf, rij, edi])
        z = Add()([z, ds])
        v = Add()([v, dv])
        # Update
        ds, dv = PAiNNUpdate(units=128)([z, v])
        z = Add()([z, ds])
        v = Add()([v, dv])
    n = z

    n_rardical = GatherNodes()([n, eri_n])
    e_radical = GatherNodesIngoing()([rbf, eri_e])

    rad_embedd = Concatenate(axis=-1)([n_rardical, e_radical])
    rad_embedd = PoolingNodes()(rad_embedd)

    main_output = MLP(**output_mlp)(rad_embedd)
    model = tf.keras.models.Model(inputs=[node_input, equiv_input, xyz_input, bond_index_input, node_radical_index, edge_radical_index],
                                  outputs=main_output)

    return model

#  Schnet model adapted for barrier energy from structure calculation
def schnet_adapted(
        # Input
        input_node_shape,
        input_edge_shape,
        input_state_shape,
        input_embedd: dict = None,
        # Output
        output_mlp: dict = None,
        output_dense: dict = None,
        output_embedd: dict = None,
        # Model specific
        depth=4,
        out_scale_pos=0,
        interaction_args: dict = None,
        node_pooling_args: dict = None
    ):
    """
    Make uncompiled SchNet model.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_embedd (list): Dictionary of embedding parameters used if input shape is None. Default is
            {'input_node_vocab': 95, 'input_edge_vocab': 5, 'input_state_vocab': 100,
            'input_node_embedd': 64, 'input_edge_embedd': 64, 'input_state_embedd': 64,
            'input_type': 'ragged'}
        output_mlp (dict, optional): Parameter for MLP output classification/ regression. Defaults to
            {"use_bias": [True, True], "units": [128, 64],
            "activation": ['shifted_softplus', 'shifted_softplus']}
        output_dense (dict): Parameter for Dense scaling layer. Defaults to {"units": 1, "activation": 'linear',
             "use_bias": True}.
        output_embedd (str): Dictionary of embedding parameters of the graph network. Default is
             {"output_mode": 'graph', "output_type": 'padded'}
        depth (int, optional): Number of Interaction units. Defaults to 4.
        out_scale_pos (int, optional): Scaling output, position of layer. Defaults to 0.
        interaction_args (dict): Interaction Layer arguments. Defaults include {"node_dim" : 128, "use_bias": True,
             "activation" : 'shifted_softplus', "cfconv_pool" : 'segment_sum',
             "is_sorted": False, "has_unconnected": True}
        node_pooling_args (dict, optional): Node pooling arguments. Defaults to {"pooling_method": "segment_sum"}.

    Returns:
        model (tf.keras.models.Model): SchNet.

    """
    # Make default values if None
    model_default = {'input_embedd': {'input_node_vocab': 95, 'input_edge_vocab': 5, 'input_state_vocab': 100,
                                      'input_node_embedd': 64, 'input_edge_embedd': 64, 'input_state_embedd': 64,
                                      },
                     'output_embedd': {"output_mode": 'graph', "output_type": 'padded'},
                     'interaction_args': {"units": 128, "use_bias": True,
                                          "activation": 'kgcnn>shifted_softplus', "cfconv_pool": 'sum',
                                          "is_sorted": False, "has_unconnected": True},
                     'output_mlp': {"use_bias": [True, True], "units": [128, 64],
                                    "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus']},
                     'output_dense': {"units": 1, "activation": 'linear', "use_bias": True},
                     'node_pooling_args': {"pooling_method": "sum"}
                     }

    # Update args
    input_embedd = update_model_args(model_default['input_embedd'], input_embedd)
    interaction_args = update_model_args(model_default['interaction_args'], interaction_args)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    output_dense = update_model_args(model_default['output_dense'], output_dense)
    output_embedd = update_model_args(model_default['output_embedd'], output_embedd)
    node_pooling_args = update_model_args(model_default['node_pooling_args'], node_pooling_args)

    # Make input embedding, if no feature dimension
    node_input, n, edge_input, ed, edge_index_input, env_input, uenv = generate_standard_graph_input(input_node_shape,
                                                                                        input_edge_shape, input_state_shape,
                                                                                        **input_embedd)
    
    node_radical_index = ks.layers.Input(shape=(None,2), name='node_radical_input', dtype="int64", ragged=True)
    eri_n = node_radical_index 
    edge_radical_index = ks.layers.Input(shape=(None, 1), name='edge_radical_input', dtype="int64", ragged=True)
    eri_e = edge_radical_index
    edi = edge_index_input

    n = Dense(interaction_args["units"], activation='linear')(n)
    for i in range(0, depth):
        n = SchNetInteraction(**interaction_args)([n, ed, edi])

    n_rardical = GatherNodes()([n, eri_n])
    e_radical = GatherNodesIngoing()([ed, eri_e])

    rad_embedd = Concatenate(axis=-1)([n_rardical, e_radical])
    rad_embedd = PoolingNodes()(rad_embedd)

    rad_embedd = MLP(**output_mlp)(rad_embedd)
    main_output = Dense(**output_dense)(rad_embedd)

    #main_output = ks.layers.Flatten()(rad_embedd)  # will be dense
    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, node_radical_index, edge_radical_index], outputs=main_output)

    return model

#  MegNet model adapted for barrier energy from structure calculation
def megnet_adapted(
        # Input
        input_node_shape,
        input_edge_shape,
        input_state_shape,
        input_embedding: dict = None,
        # Output
        output_embedding: dict = None,  # Only graph possible for megnet
        output_mlp: dict = None,
        # Model specs
        meg_block_args: dict = None,
        node_ff_args: dict = None,
        edge_ff_args: dict = None,
        state_ff_args: dict = None,
        nblocks: int = 3,
        has_ff: bool = True,
        dropout: float = None,
    ):
    """
    Make uncompiled Megnet model.

    Args:
        input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
        input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
        input_state_shape (list): Shape of state features. If shape is (None,) embedding layer is used.
        input_embedding (list): Dictionary of embedding parameters used if input shape is None. Default is
            {'input_node_vocab': 95, 'input_edge_vocab': 5, 'input_state_vocab': 100,
            'input_node_embedd': 64, 'input_edge_embedd': 64, 'input_state_embedd': 64,
            'input_type': 'ragged'}
        output_mlp (dict, optional): Parameter for MLP output classification/ regression. Defaults to
            {"use_bias": [True, True], "units": [128, 64],
            "activation": ['shifted_softplus', 'shifted_softplus']}
        output_embedding (str): Dictionary of embedding parameters of the graph network. Default is
                {"output_mode": 'graph', "output_type": 'padded'}
        nblocks (int, optional): Number of Interaction units. Defaults to 3.
        out_scale_pos (int, optional): Scaling output, position of layer. Defaults to 0.
        meg_block_args (dict): Interaction Layer arguments. Defaults include {'node_embed': [64, 32, 32], 'edge_embed': [64, 32, 32],
                                        'env_embed': [64, 32, 32], 'activation': 'kgcnn>softplus2'}
        node_ff_args (dict, optional): Dictionary of layer arguments unpacked in `MLP` feed-forward layer. Defualts include: {
            "units": [64, 32], "activation": "kgcnn>softplus2"},
        edge_ff_args(dict, optional): Dictionary of layer arguments unpacked in `MLP` feed-forward layer. Defualts include: {
            "units": [64, 32], "activation": "kgcnn>softplus2"},
        state_ff_args(dict, optional): Dictionary of layer arguments unpacked in `MLP` feed-forward layer.Defualts include: {
            "units": [64, 32], "activation": "kgcnn>softplus2"}
        has_ff (bool): Use feed-forward MLP in each block.
        dropout (int): Dropout to use. Default is None.

    Returns:
        model (tf.keras.models.Model): Megnet.

    """
    # Default arguments if None
    model_default = {'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 64},
                                        "edges": {"input_dim": 5, "output_dim": 64},
                                        "state": {"input_dim": 100, "output_dim": 64}},
                    'output_embedding': {"output_mode": 'graph', "output_tensor_type": 'padded'},
                    'output_mlp': {"use_bias": [True, True, True], "units": [128, 64, 1],
                                    "activation": ['kgcnn>swish', 'kgcnn>swish', 'linear']},
                    'meg_block_args': {'node_embed': [64, 32, 32], 'edge_embed': [64, 32, 32],
                                        'env_embed': [64, 32, 32], 'activation': 'kgcnn>softplus2'},
                    'node_ff_args': {"units": [64, 32], "activation": "kgcnn>softplus2"},
                    'edge_ff_args': {"units": [64, 32], "activation": "kgcnn>softplus2"},
                    'state_ff_args': {"units": [64, 32], "activation": "kgcnn>softplus2"}
                    }

    # Update default arguments
    input_embedding = update_model_args(model_default['input_embedding'], input_embedding)
    output_embedding = update_model_args(model_default['output_embedding'], output_embedding)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)
    meg_block_args = update_model_args(model_default['meg_block_args'], meg_block_args)
    node_ff_args = update_model_args(model_default['node_ff_args'], node_ff_args)
    edge_ff_args = update_model_args(model_default['edge_ff_args'], edge_ff_args)
    state_ff_args = update_model_args(model_default['state_ff_args'], state_ff_args)
    state_ff_args.update({"input_tensor_type": "tensor"})

    # Make input embedding, if no feature dimension
    print(input_node_shape)
    
    node_input, n, edge_input, ed, edge_index_input, env_input, uenv = generate_standard_graph_input(input_node_shape,
                                                                                        input_edge_shape, input_state_shape,
                                                                                        input_embedding['nodes'],
                                                                                        input_embedding['edges'],
                                                                                        input_embedding['state'],)
    node_radical_index = ks.layers.Input(shape=(None, 2), name='node_radical_input', dtype="int64", ragged=True)
    eri_n = node_radical_index
    edge_radical_index = ks.layers.Input(shape=(None, 1), name='edge_radical_input', dtype="int64", ragged=True)
    eri_e = edge_radical_index
    edi = edge_index_input

    # starting
    vp = n
    ep = ed
    up = uenv
    vp = MLP(**node_ff_args)(vp)
    ep = MLP(**edge_ff_args)(ep)
    up = MLP(**state_ff_args)(up)
    vp2 = vp
    ep2 = ep
    up2 = up
    for i in range(0, nblocks):
        if has_ff and i > 0:
            vp2 = MLP(**node_ff_args)(vp)
            ep2 = MLP(**edge_ff_args)(ep)
            up2 = MLP(**state_ff_args)(up)

        # MEGnetBlock
        vp2, ep2, up2 = MEGnetBlock(**meg_block_args)(
            [vp2, ep2, edi, up2])

        # skip connection
        if dropout is not None:
            vp2 = Dropout(dropout, name='dropout_atom_%d' % i)(vp2)
            ep2 = Dropout(dropout, name='dropout_bond_%d' % i)(ep2)
            up2 = Dropout(dropout, name='dropout_state_%d' % i)(up2)

        vp = Add()([vp2, vp])
        ep = Add()([ep2, ep])
        up = Add(input_tensor_type="tensor")([up2, up])


    n_rardical = GatherNodes()([vp, eri_n])
    e_radical = GatherNodesIngoing()([ep, eri_e])

    rad_embedd = Concatenate(axis=-1)([n_rardical, e_radical])
    rad_embedd = PoolingNodes()(rad_embedd)

    rad_embedd = MLP(**output_mlp)(rad_embedd)


    model = ks.models.Model(inputs=[node_input, edge_input, edge_index_input, env_input, node_radical_index, edge_radical_index], outputs=rad_embedd)

    return model

#  DimeNet++ model adapted for barrier energy from structure calculation
def dimenet_adapted(
        # Input
        input_node_shape,
        input_embedd: dict = None,
        # Output
        output_embedd: dict = None,
        output_mlp: dict = None,
        # Model specific parameter
        emb_size = 128,
        out_emb_size = 256,
        int_emb_size = 64,
        basis_emb_size =8,
        num_blocks = 4,
        num_spherical = 7,
        num_radial= 6,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_dense_output=3,
        num_targets=12,
        activation="swish",
        # extensive=True,
        output_init='zeros',
        ):
        """
        Make uncompiled Dimenet++ model.

        Args:
            input_node_shape (list): Shape of node features. If shape is (None,) embedding layer is used.
            input_edge_shape (list): Shape of edge features. If shape is (None,) embedding layer is used.
            input_state_shape (list): Shape of state features. If shape is (None,) embedding layer is used.
            input_embedd (list): Dictionary of embedding parameters used if input shape is None. Default is
                {'input_node_vocab': 95, 'input_edge_vocab': 5, 'input_state_vocab': 100,
                'input_node_embedd': 64, 'input_edge_embedd': 64, 'input_state_embedd': 64,
                'input_type': 'ragged'}
            output_mlp (dict, optional): Parameter for MLP output classification/ regression. Defaults to
                {"use_bias": [True, True], "units": [128, 64],
                "activation": ['shifted_softplus', 'shifted_softplus']}
            output_embedd (str): Dictionary of embedding parameters of the graph network. Default is
                    {"output_mode": 'graph', "output_type": 'padded'}
            emb_size (int): Overall embedding size used for the messages.
            out_emb_size (int): Embedding size for output of `DimNetOutputBlock`.
            int_emb_size (int): Embedding size used for interaction triplets.
            basis_emb_size (int): Embedding size used inside the basis transformation.
            num_blocks (int): Number of graph embedding blocks or depth of the network. Default = 4.
            num_spherical (int): Number of spherical components in `SphericalBasisLayer`. Default = 7.
            num_radial (int): Number of radial components in basis layer. Default = 6.
            cutoff (float): Distance cutoff for basis layer. Default = 5.0.
            envelope_exponent (int): Exponent in envelope function for basis layer. Default = 5.
            num_before_skip (int): Number of residual layers in interaction block before skip connection. Default = 1.
            num_after_skip (int): Number of residual layers in interaction block after skip connection. Default = 2.
            num_dense_output (int): Number of dense units in output `DimNetOutputBlock`. Default = 3.
            num_targets (int): Number of targets or output embedding dimension of the model. Default = 12.
            activation (str, dict): Activation to use.
            extensive (bool): Graph output for extensive target to apply sum for pooling or mean otherwise.
            output_init (str, dict): Output initializer for kernel.

        Returns:
            model (tf.keras.models.Model): Megnet.

        """

    model_default = {'input_embedd': {'input_node_vocab': 95, 'input_node_embedd': 64, 'input_tensor_type': 'ragged'},
                        'output_mlp': {"use_bias": [True, True, True], "units": [128, 64, 1],
                                    "activation": ['kgcnn>swish', 'kgcnn>swish', 'linear']},
                        }

    input_embedd = update_model_args(model_default['input_embedd'], input_embedd)
    output_mlp = update_model_args(model_default['output_mlp'], output_mlp)

    node_input, n, xyz_input, bond_index_input, angle_index_input, _ = generate_mol_graph_input(input_node_shape,
                                                                                                [None, 3],
                                                                                                [None, 2],
                                                                                                [None, 2],
                                                                                                **input_embedd)
    node_radical_index = ks.layers.Input(shape=(None,2), name='node_radical_input', dtype="int64", ragged=True)
    eri_n = node_radical_index
    edge_radical_index = ks.layers.Input(shape=(None, 1), name='edge_radical_input', dtype="int64", ragged=True)
    eri_e = edge_radical_index

    x = xyz_input
    edi = bond_index_input
    adi = angle_index_input

    # Calculate distances
    d = NodeDistance()([x, edi])
    rbf = BesselBasisLayer(num_radial=num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent)(d)

    # Calculate angles
    a = EdgeAngle()([x, edi, adi])
    sbf = SphericalBasisLayer(num_spherical=num_spherical, num_radial=num_radial, cutoff=cutoff,
                                envelope_exponent=envelope_exponent)([d, a, adi])

    # Embedding block
    rbf_emb = Dense(emb_size, use_bias=True, activation=activation, kernel_initializer="orthogonal")(rbf)
    n_pairs = GatherNodes()([n, edi])
    x = Concatenate(axis=-1)([n_pairs, rbf_emb])
    x = Dense(emb_size, use_bias=True, activation=activation, kernel_initializer="orthogonal")(x)
    ps = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=num_targets,
                            output_kernel_initializer=output_init)([n, x, rbf, edi])

    # Interaction blocks
    add_xp = Add()
    for i in range(num_blocks):
        x = DimNetInteractionPPBlock(emb_size, int_emb_size, basis_emb_size, num_before_skip, num_after_skip)(
            [x, rbf, sbf, adi])
        p_update = DimNetOutputBlock(emb_size, out_emb_size, num_dense_output, num_targets=num_targets,
                                        output_kernel_initializer=output_init)([n, x, rbf, edi])
        ps = add_xp([ps, p_update])

    n_rardical = GatherNodes()([ps, eri_n])
    e_radical = GatherNodesIngoing()([x, eri_e])

    rad_embedd = Concatenate(axis=-1)([n_rardical, e_radical])
    rad_embedd = PoolingNodes()(rad_embedd)

    main_output = MLP(**output_mlp)(rad_embedd)


    model = tf.keras.models.Model(inputs=[node_input, xyz_input, bond_index_input, angle_index_input, node_radical_index, edge_radical_index],
                                    outputs=main_output)

    return model

# Generates PaiNN model with optmal paramters
def make_painn(optimizer, metric):
    """Get PaiNN model for HAT.
    Args:
        **kwargs
    Returns:
        tf.keras.models.Model: PaiNN keras model.
    """
    #Defining the model - here the optimal parameters
    model = painn_adapted(input_node_shape=[None],
                        input_equiv_shape=[None, 128, 3],
                        input_embedding= {"nodes": {"input_dim": 95, "output_dim": 128}},
                        depth=3
                        )
    #Compiling the final model
    model.compile(loss='mean_absolute_error',
                    optimizer=optimizer,
                    metrics=[metric])
    return(model)

# Generates Schnet model with optmal paramters
def make_schnet(optimizer, metric):
    """Get Schnet model for HAT.
    Args:
        **kwargs
    Returns:
        tf.keras.models.Model: Schnet keras model.
    """
        #Defining the model - here the optimal parameters
    model = schnet_adapted(
                #Input
                input_node_shape = [None],
                input_edge_shape = [None,40],
                input_state_shape = [40],
                input_embedd={'input_node_vocab': 20,
                            'input_node_embedd': 128,
                            'input_tensor_type': 'ragged'},
                # Output
                output_mlp={"use_bias": True,
                            "units": [128, 64],
                            "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus']},
                output_dense={"units": 1, "activation": 'linear', "use_bias": True},
                output_embedd={'output_mode': 'graph'},
                # Model specific
                depth=4,
                interaction_args={"units": 128,
                                "use_bias": True,
                                "activation": 'kgcnn>shifted_softplus',
                                "cfconv_pool": "segment_sum",
                                "is_sorted": False,
                                "has_unconnected": True
                                },
                node_pooling_args = {"pooling_method": "mean"}
            )
    #Compiling the final model
    model.compile(loss='mean_absolute_error',
                    optimizer=optimizer,
                    metrics=[metric])
    return(model)

# Generates MegNet model with optmal paramters
def make_megnet(optimizer, metric):
    """Get MegNet model for HAT.
    Args:
        **kwargs
    Returns:
        tf.keras.models.Model: Megnet keras model.
    """
        #Defining the model - here the optimal parameters
    model = megnet_adapted(
            # Input
            input_node_shape=[None],
            input_edge_shape=[None, 40],
            input_state_shape=[1],
            # Output
            output_embedding={"output_mode": 'graph', "output_type": 'padded'},  # Only graph possible for megnet
            output_mlp={"use_bias": [True, True, True],
                    "units": [128, 64, 1],
                    "activation": ['kgcnn>shifted_softplus', 'kgcnn>shifted_softplus', 'linear']},
            # Model specs
            meg_block_args={'node_embed': [64, 32, 32], 'edge_embed': [64, 32, 32], 'env_embed': [64, 32, 32],
                        'activation': 'kgcnn>shifted_softplus', 'is_sorted': False},
            nblocks=3,
            )
    #Compiling the final model
    model.compile(loss='mean_absolute_error',
                    optimizer=optimizer,
                    metrics=[metric])
    return(model)

# Generates DimeNet++ model with optmal paramters
def make_dimenet(optimizer, metric):
    """Get Dimenet++ model for HAT.
    Args:
        **kwargs
    Returns:
        tf.keras.models.Model: Dimenet++ keras model.
    """
        #Defining the model - here the optimal parameters
    model = dimenet_adapted(
        # Input
        input_node_shape=[None],
        input_embedd={'input_node_vocab': 20,
                    'input_node_embedd': 128,
                    'input_tensor_type': 'ragged'},
        # Output
        output_embedd=None,
        output_mlp={"use_bias": True,
                    "units": [128, 64, 1],
                    "activation": ['kgcnn>swish', 'kgcnn>swish', 'linear']},
        # Model specific parameter
        emb_size=128,
        out_emb_size=256,
        int_emb_size=64,
        basis_emb_size=8,
        num_blocks=4,
        num_spherical=7,
        num_radial=6,
        cutoff=5.0,
        envelope_exponent=5,
        num_before_skip=1,
        num_after_skip=2,
        num_dense_output=3,
        num_targets=12,
        activation="swish",
        # extensive=True,
        output_init='zeros',
    )
    #Compiling the final model
    model.compile(loss='mean_absolute_error',
                    optimizer=optimizer,
                    metrics=[metric])
    return(model)

# Non-optimized PaiNN model from the kgcnn example (https://github.com/aimat-lab/gcnn_keras)
def make_painn_noopt(**kwargs):
    """Get PAiNN model.
    Args:
        **kwargs
    Returns:
        tf.keras.models.Model: PAiNN keras model.
    """
    model_args = kwargs
    model_default = {'input_node_shape': None, 'input_equiv_shape': None,
                     'input_embedding': {"nodes": {"input_dim": 95, "output_dim": 128}},
                     'output_embedding': {"output_mode": 'graph', "output_tensor_type": 'padded'},
                     'output_mlp': {"use_bias": [True, True], "units": [128, 1],
                                    "activation": ['swish', 'linear']},
                     'bessel_basis': {'num_radial': 20, 'cutoff': 5.0, 'envelope_exponent': 5},
                     'pooling_args': {'pooling_method': 'sum'},
                     'depth': 3,
                     'verbose': 1
                     }
    m = update_model_args(model_default, model_args)
    if m['verbose'] > 0:
        print("INFO: Updated functional make model kwargs:")
        pprint.pprint(m)

    # local variables
    input_node_shape = m['input_node_shape']
    input_equiv_shape = m['input_equiv_shape']
    input_embedding = m['input_embedding']
    bessel_basis = m['bessel_basis']
    depth = m['depth']
    output_embedding = m['output_embedding']
    pooling_args = m['pooling_args']
    output_mlp = m['output_mlp']

    # Make input
    node_input = ks.layers.Input(shape=input_node_shape, name='node_input', dtype="float32", ragged=True)
    equiv_input = ks.layers.Input(shape=input_equiv_shape, name='equiv_input', dtype="float32", ragged=True)
    xyz_input = ks.layers.Input(shape=[None, 3], name='xyz_input', dtype="float32", ragged=True)
    bond_index_input = ks.layers.Input(shape=[None, 2], name='bond_index_input', dtype="int64", ragged=True)

    # Embedding
    z = generate_node_embedding(node_input, input_node_shape, input_embedding['nodes'])
    edi = bond_index_input
    x = xyz_input
    v = equiv_input

    rij = EdgeDirectionNormalized()([x, edi])
    d = NodeDistance()([x, edi])
    rbf = BesselBasisLayer(**bessel_basis)(d)

    for i in range(depth):
        # Message
        ds, dv = PAiNNconv(units=128)([z, v, rbf, rij, edi])
        z = Add()([z, ds])
        v = Add()([v, dv])
        # Update
        ds, dv = PAiNNUpdate(units=128)([z, v])
        z = Add()([z, ds])
        v = Add()([v, dv])
    n = z
    # Output embedding choice
    if output_embedding["output_mode"] == 'graph':
        out = PoolingNodes(**pooling_args)(n)
        main_output = MLP(**output_mlp)(out)
    else:  # Node labeling
        out = n
        main_output = MLP(**output_mlp)(out)
        main_output = ChangeTensorType(input_tensor_type="ragged", output_tensor_type="tensor")(main_output)
        # no ragged for distribution atm

    model = tf.keras.models.Model(inputs=[node_input, equiv_input, xyz_input, bond_index_input],
                                  outputs=main_output)

    return model
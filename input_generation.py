from numpy.core.numeric import outer
import tensorflow as tf
import os
import numpy as np
import kgcnn
from kgcnn.utils.adj import coordinates_to_distancematrix,define_adjacency_from_distance, distance_to_gauss_basis, get_angle_indices, sort_edge_indices
from kgcnn.utils.data import save_pickle_file,load_pickle_file, ragged_tensor_from_nested_numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ase.io import read
from utils import get_distance


def generate_input_data(input_dir, output_dir):
    """
    Generates nodes, edges, coordinates and distances lists from .pdb files
    Args: 
        input_dir (str): direction for the .pdb files
        output_dir (str): output direction
    """

    search_path = input_dir
    filelist = [f for f in os.listdir(search_path) if os.path.isfile(os.path.join(search_path, f))]
    base_list = [x.split(".")[0].split("_")[0] + "_" + x.split(".")[0].split("_")[1] for x in filelist]
    base_set = sorted(set(base_list))


    all_nodes = []
    all_edges = []
    all_labels = []
    all_atoms = []
    for x in base_set:
        mol1 = read(search_path+x+"_1.pdb")
        an1 = mol1.get_atomic_numbers()
        n_1 = mol1.get_global_number_of_atoms()
        pos1 = mol1.positions
        mol2 = read(search_path+x+"_2.pdb")
        n_2 = mol2.get_global_number_of_atoms()
        an2 = mol2.get_atomic_numbers()
        pos2 = mol2.positions
        nodes1 = np.concatenate((np.array([0]),an1),axis=0)
        edges1 = np.concatenate((np.array([pos2[0]]),pos1),axis=0)
        nodes2 = np.concatenate((np.array([0]),an2),axis=0)
        edges2 = np.concatenate((np.array([pos1[0]]),pos2),axis=0)
        arr = np.load(search_path+x+".npz", allow_pickle=True)
        arr = np.expand_dims(arr[arr.files[0]], axis=0)[0]
        if 'e_max' in arr and 'e_00' in arr and 'e_10' in arr:
            lab1 = arr['e_max'] - arr['e_00']
            lab2 = arr['e_max'] - arr['e_10']
            all_labels.append(lab1)
            all_labels.append(lab2)
            all_nodes.append(nodes1)
            all_nodes.append(nodes2)
            all_edges.append(edges1)
            all_edges.append(edges2)
            all_atoms.append(n_1)
            all_atoms.append(n_2)


    dist = [coordinates_to_distancematrix(x) for x in all_edges]
    adj = [define_adjacency_from_distance(x,max_distance=8, max_neighbours=25)[0] for x in dist]
    edge_indices = [define_adjacency_from_distance(x,max_distance=8, max_neighbours=25)[1] for x in dist]
    edge_dist = [distance_to_gauss_basis(dist[i][adj[i]], bins=40, distance=8, sigma=0.4) for i in range(len(adj))]


    save_pickle_file(all_edges, output_dir + "cr_coord.pickle")
    save_pickle_file(all_nodes, output_dir + "cr_nodes.pickle",)
    save_pickle_file(edge_dist, output_dir + "cr_edges.pickle",)
    save_pickle_file(edge_indices, output_dir + "cr_indices.pickle")
    np.save(output_dir + "cr_atoms.npy", all_atoms)
    np.save(output_dir + "cr_labels.npy",all_labels)


def generate_input_data_cutoff(input_dir, output_dir, cutoff_dist):
    """
    Generates nodes, edges, coordinates and distances lists from .pdb files with a cutoff distance
    Args: 
        input_dir (str): direction for the .pdb files
        output_dir (str): output direction
        cutoff_dist (int): distance for cutoff (in Angsroem)
    """
    search_path = input_dir
    filelist = [f for f in os.listdir(search_path) if os.path.isfile(os.path.join(search_path, f))]
    base_list = [x.split(".")[0].split("_")[0] + "_" + x.split(".")[0].split("_")[1] for x in filelist]
    base_set = sorted(set(base_list))

    threshold = cutoff_dist

    all_nodes = []
    all_edges = []
    all_labels = []
    all_atoms = []
    all_nodes_high = []
    all_edges_high = []
    all_labels_high = []
    all_atoms_high = []
    for x in base_set:
        mol1 = read(search_path+x+"_1.pdb")
        an1 = mol1.get_atomic_numbers()
        pos1 = mol1.positions
        n_1 = mol1.get_global_number_of_atoms()
        mol2 = read(search_path+x+"_2.pdb")
        an2 = mol2.get_atomic_numbers()
        pos2 = mol2.positions
        n_2 = mol2.get_global_number_of_atoms()
        nodes1 = np.concatenate((np.array([0]),an1),axis=0)
        edges1 = np.concatenate((np.array([pos2[0]]),pos1),axis=0)
        nodes2 = np.concatenate((np.array([0]),an2),axis=0)
        edges2 = np.concatenate((np.array([pos1[0]]),pos2),axis=0)
        dist = get_distance(edges1[0,:], edges2[0,:])

        arr = np.load(search_path+x+".npz", allow_pickle=True)
        arr = np.expand_dims(arr[arr.files[0]], axis=0)[0]
        if 'e_max' in arr and 'e_00' in arr and 'e_10' in arr:
            lab1 = arr['e_max'] - arr['e_00']
            lab2 = arr['e_max'] - arr['e_10']
            if dist < threshold: 
                all_labels.append(lab1)
                all_labels.append(lab2)
                all_nodes.append(nodes1)
                all_nodes.append(nodes2)
                all_edges.append(edges1)
                all_edges.append(edges2)
                all_atoms.append(n_1)
                all_atoms.append(n_2)
            else:
                all_nodes_high.append(nodes1)
                all_nodes_high.append(nodes2)
                all_labels_high.append(lab1)
                all_labels_high.append(lab2)
                all_edges_high.append(edges1)
                all_edges_high.append(edges2)
                all_atoms_high.append(n_1)
                all_atoms_high.append(n_2)


    dist = [coordinates_to_distancematrix(x) for x in all_edges]
    adj = [define_adjacency_from_distance(x,max_distance=3, max_neighbours=25)[0] for x in dist]
    edge_indices = [define_adjacency_from_distance(x,max_distance=3, max_neighbours=25)[1] for x in dist]
    edge_dist = [distance_to_gauss_basis(dist[i][adj[i]], bins=40, distance=8, sigma=0.4) for i in range(len(adj))]

    dist = [coordinates_to_distancematrix(x) for x in all_edges_high]
    adj = [define_adjacency_from_distance(x,max_distance=3, max_neighbours=25)[0] for x in dist]
    edge_indices_high = [define_adjacency_from_distance(x,max_distance=3, max_neighbours=25)[1] for x in dist]
    edge_dist_high = [distance_to_gauss_basis(dist[i][adj[i]], bins=40, distance=8, sigma=0.4) for i in range(len(adj))]

    save_pickle_file(all_edges, output_dir+str(cutoff_dist)+"_low_cr_coord.pickle")
    save_pickle_file(all_nodes, output_dir+str(cutoff_dist)+"_low_cr_nodes.pickle",)
    save_pickle_file(edge_dist, output_dir+str(cutoff_dist)+"_low_cr_edges.pickle",)
    save_pickle_file(edge_indices, output_dir+str(cutoff_dist)+"_low_cr_indices.pickle")
    np.save(output_dir+str(cutoff_dist)+"_low_cr_labels.npy",all_labels)
    np.save(output_dir+str(cutoff_dist)+"_low_cr_atoms.npy",all_atoms)

    save_pickle_file(all_edges_high, output_dir+str(cutoff_dist)+"_high_cr_coord.pickle")
    save_pickle_file(all_nodes_high, output_dir+str(cutoff_dist)+"_high_cr_nodes.pickle",)
    save_pickle_file(edge_dist_high, output_dir+str(cutoff_dist)+"_high_cr_edges.pickle",)
    save_pickle_file(edge_indices_high, output_dir+str(cutoff_dist)+"_high_cr_indices.pickle")
    np.save(output_dir+str(cutoff_dist)+"_high_cr_labels.npy",all_labels_high)
    np.save(output_dir+str(cutoff_dist)+"_high_cr_atoms.npy",all_atoms_high)

def painn_dataset(input_dir, split_size):
    """Creates PaiNN graph input
    Args: 
        input_dir (str): input direction with files created using generate_input_data function
        split_size (int): test size 
    Return:
        xtrain (ragged tensor): training data 
        xtest (ragged tensor): test data
        xval (ragged tensor): validation data 
        ytrain (ragged tensor): training labels
        ytest (ragged tensor): test labels
        yval (ragged tensor): validation labels
        scaler: scaler config for later rescaling of the data
        """
    
    # load Dataset
    y_data = np.expand_dims(np.load(input_dir+"cr_labels.npy"),axis=-1)
    nodes = load_pickle_file(input_dir+"cr_nodes.pickle")
    coord = load_pickle_file(input_dir+"cr_coord.pickle")

    equivariant = [np.zeros((len(x), 128, 3)) for x in nodes]

    dist = [coordinates_to_distancematrix(x) for x in coord]
    edge_indices = [define_adjacency_from_distance(x, max_distance=5, max_neighbours=25)[1] for x in dist]
    edge_indices = [x if x[0,1]==1 else sort_edge_indices(np.concatenate([np.array([[0, 1], [1,0]]), x], axis=0)) for x in edge_indices]
    del dist  # Frees some memory

    # Should check again the data function to ensure that radical and hydrogen are on top an edge between them exists.
    radical_node_index = [np.array([[0, 1]]) for _ in edge_indices]  
    radical_edge_index = [np.array([[0]]) for _ in radical_node_index]

    scaler = StandardScaler(copy=True)
    labels = scaler.fit_transform(y_data)
    
    # Train Test split
    labels_new, labels_val, nodes_new, nodes_val, equi_new, equi_val, coord_new, coord_val, \
        edge_indices_new, edge_indices_val, radical_node_index_new, radical_node_index_val, radical_edge_index_new, radical_edge_index_val= train_test_split(labels,
        nodes, equivariant, coord, edge_indices, radical_node_index, radical_edge_index, test_size=split_size)
    del labels, nodes, equivariant, coord, edge_indices, radical_node_index , radical_edge_index,

    # Train Test split
    labels_train, labels_test, nodes_train, nodes_test, equi_train, equi_test, coord_train, coord_test, \
        edge_indices_train, edge_indices_test, radical_node_index_train, radical_node_index_test, radical_edge_index_train, \
            radical_edge_index_test= train_test_split( labels_new,
        nodes_new, equi_new, coord_new, edge_indices_new, radical_node_index_new, radical_edge_index_new, test_size=split_size)
    del labels_new, nodes_new, equi_new, coord_new, edge_indices_new 

    # Convert to tf.RaggedTensor or tf.tensor
    # a copy of the data is generated by ragged_tensor_from_nested_numpy()
    nodes_train, equi_train, coord_train, edge_indices_train, radical_node_index_train, radical_edge_index_train = ragged_tensor_from_nested_numpy(
        nodes_train), ragged_tensor_from_nested_numpy(equi_train), ragged_tensor_from_nested_numpy(coord_train), ragged_tensor_from_nested_numpy(
        edge_indices_train), ragged_tensor_from_nested_numpy(radical_node_index_train), ragged_tensor_from_nested_numpy(radical_edge_index_train)

    nodes_test, equi_test, coord_test, edge_indices_test, radical_node_index_test, radical_edge_index_test = ragged_tensor_from_nested_numpy(
        nodes_test),  ragged_tensor_from_nested_numpy(equi_test), ragged_tensor_from_nested_numpy(coord_test), ragged_tensor_from_nested_numpy(
        edge_indices_test), ragged_tensor_from_nested_numpy(radical_node_index_test), ragged_tensor_from_nested_numpy(radical_edge_index_test)

    nodes_val, equi_val, coord_val, edge_indices_val, radical_node_index_val, radical_edge_index_val = ragged_tensor_from_nested_numpy(
        nodes_val),  ragged_tensor_from_nested_numpy(equi_val), ragged_tensor_from_nested_numpy(coord_val), ragged_tensor_from_nested_numpy(
        edge_indices_val), ragged_tensor_from_nested_numpy(radical_node_index_val), ragged_tensor_from_nested_numpy(radical_edge_index_val)

    # Define input and output data
    xtrain = nodes_train, equi_train, coord_train, edge_indices_train, radical_node_index_train, radical_edge_index_train
    xtest = nodes_test, equi_test, coord_test, edge_indices_test, radical_node_index_test, radical_edge_index_test
    ytrain = labels_train
    ytest = labels_test
    xval = nodes_val, equi_val, coord_val, edge_indices_val, radical_node_index_val, radical_edge_index_val
    yval = labels_val
    return(xtrain, xtest, xval, ytrain, ytest, yval, scaler)


def schnet_dataset(input_dir, split_size):
    """
    Creates Schnet graph input
    Args: 
        input_dir (str): input direction with files created using generate_input_data function
        split_size (int): test size 
    Return:
        xtrain (ragged tensor): training data 
        xtest (ragged tensor): test data
        xval (ragged tensor): validation data 
        ytrain (ragged tensor): training labels
        ytest (ragged tensor): test labels
        yval (ragged tensor): validation labels
        scaler: scaler config for later rescaling of the data
    """

    y_data = np.expand_dims(np.load(input_dir+"cr_labels.npy"),axis=-1)
    nodes = load_pickle_file(input_dir+'cr_nodes.pickle')
    edges = load_pickle_file(input_dir+"cr_edges.pickle")
    edge_indices = load_pickle_file(input_dir+"cr_indices.pickle")

    radical_node_index = [np.expand_dims(x[0],axis=0) for x in edge_indices] 
    radical_edge_index = [np.array([[0]]) for _ in radical_node_index]

    scaler = StandardScaler(copy=True)
    labels = scaler.fit_transform(y_data)

    # Train Test split
    labels_train, labels_test, nodes_train, nodes_test, edges_train, edges_test, edge_indices_train, edge_indices_test, \
    radical_node_index_train, radical_node_index_test, radical_edge_index_train, radical_edge_index_test = train_test_split(
        labels, nodes, edges, edge_indices, radical_node_index, radical_edge_index, test_size=split_size)
    del labels, nodes, edges, edge_indices, radical_node_index, radical_edge_index # Free memory after split, if possible

    # Convert to tf.RaggedTensor or tf.tensor
    # a copy of the data is generated by ragged_tensor_from_nested_numpy()
    nodes_train, edges_train, edge_indices_train, radical_node_index_train, radical_edge_index_train = ragged_tensor_from_nested_numpy(
        nodes_train), ragged_tensor_from_nested_numpy(edges_train), ragged_tensor_from_nested_numpy(
        edge_indices_train), ragged_tensor_from_nested_numpy(radical_node_index_train), ragged_tensor_from_nested_numpy(radical_edge_index_train)

    nodes_test, edges_test, edge_indices_test, radical_node_index_test, radical_edge_index_test = ragged_tensor_from_nested_numpy(
        nodes_test), ragged_tensor_from_nested_numpy(edges_test), ragged_tensor_from_nested_numpy(
        edge_indices_test), ragged_tensor_from_nested_numpy(radical_node_index_test), ragged_tensor_from_nested_numpy(radical_edge_index_test)

    # Define input and output data
    xtrain = nodes_train, edges_train, edge_indices_train, radical_node_index_train, radical_edge_index_train
    xval = nodes_test, edges_test, edge_indices_test, radical_node_index_test, radical_edge_index_test
    ytrain = labels_train
    yval = labels_test

    return(xtrain,xval,ytrain,yval,scaler)


def megnet_dataset(input_dir, split_size):
    """
    Creates MegNet graph input
    Args: 
        input_dir (str): input direction with files created using generate_input_data function
        split_size (int): test size 
    Return:
        xtrain (ragged tensor): training data 
        xtest (ragged tensor): test data
        xval (ragged tensor): validation data 
        ytrain (ragged tensor): training labels
        ytest (ragged tensor): test labels
        yval (ragged tensor): validation labels
        scaler: scaler config for later rescaling of the data
    """
    y_data = np.expand_dims(np.load(input_dir+"cr_labels.npy"),axis=-1)
    nodes = load_pickle_file(input_dir+"cr_nodes.pickle")
    edges = load_pickle_file(input_dir+"cr_edges.pickle")
    edge_indices = load_pickle_file(input_dir+"cr_indices.pickle")
    atoms = np.expand_dims(np.load(input_dir+"cr_atoms.npy"), axis=-1)

    radical_node_index = [np.array([[0, 1]]) for _ in edge_indices]  
    radical_edge_index = [np.array([[0]]) for _ in radical_node_index]

    scaler = StandardScaler(copy=True)
    labels = scaler.fit_transform(y_data)

    # Train Test split
    labels_train, labels_test, nodes_train, nodes_test, edges_train, edges_test, edge_indices_train, edge_indices_test, \
    atoms_train, atoms_test, radical_node_index_train, radical_node_index_test, radical_edge_index_train, \
        radical_edge_index_test = train_test_split(
        labels, nodes, edges, edge_indices, atoms, radical_node_index, radical_edge_index, test_size=split_size)
    del labels, nodes, edges, edge_indices, atoms, radical_node_index, radical_edge_index  # Free memory after split, if possible

    # Convert to tf.RaggedTensor or tf.tensor
    # a copy of the data is generated by ragged_tensor_from_nested_numpy()
    nodes_train, edges_train, edge_indices_train, atoms_train, radical_node_index_train, radical_edge_index_train = ragged_tensor_from_nested_numpy(
        nodes_train), ragged_tensor_from_nested_numpy(edges_train), ragged_tensor_from_nested_numpy(
        edge_indices_train), tf.constant(atoms_train), ragged_tensor_from_nested_numpy(
        radical_node_index_train), ragged_tensor_from_nested_numpy(radical_edge_index_train)

    nodes_test, edges_test, edge_indices_test, atoms_test, radical_node_index_test, radical_edge_index_test = ragged_tensor_from_nested_numpy(
        nodes_test), ragged_tensor_from_nested_numpy(edges_test), ragged_tensor_from_nested_numpy(
        edge_indices_test), tf.constant(atoms_test), ragged_tensor_from_nested_numpy(
        radical_node_index_test), ragged_tensor_from_nested_numpy(radical_edge_index_test)

    # Define input and output data
    xtrain = nodes_train, edges_train, edge_indices_train, atoms_train, radical_node_index_train, radical_edge_index_train
    xval = nodes_test, edges_test, edge_indices_test, atoms_test, radical_node_index_test, radical_edge_index_test
    ytrain = labels_train
    yval = labels_test
    return(xtrain,xval,ytrain,yval,scaler)


def dimenet_dataset(input_dir, split_size):
    """
    Creates DimeNet++ graph input
    Args: 
        input_dir (str): input direction with files created using generate_input_data function
        split_size (int): test size 
    Return:
        xtrain (ragged tensor): training data 
        xtest (ragged tensor): test data
        xval (ragged tensor): validation data 
        ytrain (ragged tensor): training labels
        ytest (ragged tensor): test labels
        yval (ragged tensor): validation labels
        scaler: scaler config for later rescaling of the data
    """
    y_data = np.expand_dims(np.load(input_dir+"cr_labels.npy"), axis=-1)
    nodes = load_pickle_file(input_dir+"cr_nodes.pickle")
    coord = load_pickle_file(input_dir+"cr_coord.pickle")

    dist = [coordinates_to_distancematrix(x) for x in coord]
    edge_indices = [define_adjacency_from_distance(x, max_distance=5, max_neighbours=25)[1] for x in dist]
    edge_indices = [x if x[0,1]==1 else sort_edge_indices(np.concatenate([np.array([[0, 1], [1,0]]), x], axis=0)) for x in edge_indices]
    angle_indices = [get_angle_indices(x, is_sorted=True)[2] for x in edge_indices]
    del dist  # Frees some memory

    radical_node_index = [np.array([[0, 1]]) for _ in edge_indices]  # für Kohlenstoff
    radical_edge_index = [np.array([[0]]) for _ in radical_node_index]

    scaler = StandardScaler(copy=True)
    labels = scaler.fit_transform(y_data)

    # Train Test split
    labels_train, labels_test, nodes_train, nodes_test, coord_train, coord_test, edge_indices_train, edge_indices_test, \
    angle_indices_train, angle_indices_test, \
    radical_node_index_train, radical_node_index_test, radical_edge_index_train, radical_edge_index_test = train_test_split(
        labels, nodes, coord, edge_indices, angle_indices, radical_node_index, radical_edge_index, test_size=split_size)
    del labels, nodes, coord, edge_indices, angle_indices, radical_node_index, radical_edge_index  # Free memory after split, if possible

    # Convert to tf.RaggedTensor or tf.tensor
    # a copy of the data is generated by ragged_tensor_from_nested_numpy()
    nodes_train, coord_train, edge_indices_train, angle_indices_train, radical_node_index_train, radical_edge_index_train = ragged_tensor_from_nested_numpy(
        nodes_train), ragged_tensor_from_nested_numpy(coord_train), ragged_tensor_from_nested_numpy(
        edge_indices_train), ragged_tensor_from_nested_numpy(angle_indices_train), ragged_tensor_from_nested_numpy(
        radical_node_index_train), ragged_tensor_from_nested_numpy(radical_edge_index_train)

    nodes_test, coord_test, edge_indices_test, angle_indices_test, radical_node_index_test, radical_edge_index_test = ragged_tensor_from_nested_numpy(
        nodes_test), ragged_tensor_from_nested_numpy(coord_test), ragged_tensor_from_nested_numpy(
        edge_indices_test), ragged_tensor_from_nested_numpy(angle_indices_test), ragged_tensor_from_nested_numpy(
        radical_node_index_test), ragged_tensor_from_nested_numpy(radical_edge_index_test)

    # Define input and output data
    xtrain = nodes_train, coord_train, edge_indices_train, angle_indices_train, radical_node_index_train, radical_edge_index_train
    xval = nodes_test, coord_test, edge_indices_test, angle_indices_test, radical_node_index_test, radical_edge_index_test
    ytrain = labels_train
    yval = labels_test
    return(xtrain,xval,ytrain,yval,scaler)




import networkx as nx
import numpy as np
import scipy.io as sio
import os
from multiprocessing import Pool
import argparse
import pickle


def gaussian_kernel(attribute_1, attribute_2, gamma=1):
    ''' Return the value of a gaussian kernel between two attributes '''
    return np.exp(-gamma * np.power(np.linalg.norm(attribute_1 - attribute_2), 2))


def kernel(attribute_1, attribute_2, kernel_arg):
    """ Calculate the kernel of the two attributes given
        parameters in kernel_arg
    """

    kernel_type, kernel_dict = kernel_arg

    if kernel_type == "gaussian":
        if kernel_dict["attribute_type"] == "coord":
            return gaussian_kernel(attribute_1, attribute_2, gamma=kernel_dict["gaussian_gamma_coord"])
        elif kernel_dict["attribute_type"] == "geodesic":
            return gaussian_kernel(attribute_1, attribute_2, gamma=kernel_dict["gaussian_gamma_geodesic"])
    else:
        print("problème")
        return 0

def compute_median_distances_all_pair(list_vectors, distance_type="euclidean", radius=100):
    """ given a list of vectors, compute the median of the
        distances of all pair of values in the list """

    list_distances = []
    for i, elem_i in enumerate(list_vectors):
        for j in range(i+1, len(list_vectors)):
            elem_j = list_vectors[j]
            if distance_type == "euclidean":
                distance = np.linalg.norm(elem_i - elem_j)
            elif distance_type == "geodesic":
                distance = radius * np.arccos(np.clip(np.dot(elem_i,elem_j) / np.power(radius,2),-1, 1))
                #if distance == 0:
                #    print(elem_i, elem_j)
            else:
                print("distance type not allowed !", distance_type)

            list_distances.append(distance)

    return np.median(list_distances)


def compute_heuristic_gamma(graph_1, graph_2):
    """ Calculate a gamma value for the gaussian kernel
        based on a heuristic (take the median of all
        distances between pair of coordinates)
    """

    # calcule the heuristic for graph_1

    # we get all the coordinates for graph_1
    full_coordinates = []
    for node in graph_1.nodes:
        if not graph_1.nodes[node]["is_dummy"]:
            full_coordinates.append(graph_1.nodes[node]["sphere_3dcoords"])
    graph_1_gamma_coord = compute_median_distances_all_pair(full_coordinates, "geodesic")

    # we get all the coordinates for graph_2
    full_coordinates = []
    for node in graph_2.nodes:
        if not graph_2.nodes[node]["is_dummy"]:
            full_coordinates.append(graph_2.nodes[node]["sphere_3dcoords"])
    graph_2_gamma_coord = compute_median_distances_all_pair(full_coordinates, "geodesic")

    gamma_coord = 1/np.mean([graph_1_gamma_coord, graph_2_gamma_coord])


    # we get all the geodesic distances for graph_1
    full_coordinates = []
    for edge in graph_1.edges:
        full_coordinates.append(graph_1.edges[edge]["geodesic_distance"])
    graph_1_gamma_geo = compute_median_distances_all_pair(full_coordinates, "euclidean")

    # we get all the geodesic for graph_2
    full_coordinates = []
    for edge in graph_2.edges:
        full_coordinates.append(graph_2.edges[edge]["geodesic_distance"])
    graph_2_gamma_geo = compute_median_distances_all_pair(full_coordinates, "euclidean")

    gamma_geodesic = 1/np.mean([graph_1_gamma_geo, graph_2_gamma_geo])
        
    return gamma_coord, gamma_geodesic







def full_affinity(graph_1, graph_2, kernel_args):
    """ Calculation of the affinity value of two graphs of same size with the kernel function provided
    """
    print("full affinity matrix :",graph_1.number_of_nodes(), graph_2.number_of_nodes(), graph_1.number_of_edges(), graph_2.number_of_edges())
    
    # Initialise affinity matrix with zeros
    affinity_matrix = np.zeros((np.power(graph_1.number_of_nodes(), 2), np.power(graph_2.number_of_nodes(),2)))
    
    # we fill the affinity matrix with the kernel values.
    
    # we loop over all the possible permutations
    for node_a in graph_1.nodes:
        for node_i in graph_2.nodes:
            for node_b in graph_1.nodes:
                for node_j in graph_2.nodes:

                    # Check that there s no dummy nodes
                    dummies = [graph_1.nodes[node_a]["is_dummy"],
                               graph_1.nodes[node_b]["is_dummy"],
                               graph_2.nodes[node_i]["is_dummy"],
                               graph_2.nodes[node_j]["is_dummy"]]
                    
                               
                    if not True in dummies:
                    
                        # We check if we need to take the attributes of nodes or edge
                        if node_a == node_b and node_i == node_j:
                            # We take the node attributes.
                            attribute_1 = graph_1.nodes[node_a]["sphere_3dcoords"]
                            attribute_2 = graph_2.nodes[node_i]["sphere_3dcoords"]

                            # calculate the kernel value of these attributes
                            kernel_args[1]["attribute_type"] = "coord"
                            value_kernel = kernel(attribute_1, attribute_2, kernel_args)

                            # add this in the right place in the affinity_matrix
                            affinity_matrix[ node_a * graph_2.number_of_nodes() + node_i, node_b * graph_2.number_of_nodes() + node_j] \
                                = value_kernel


                        else:
                            # we check that the edges exist on both side and if so add the value to the affinity matrix
                            if (node_a, node_b) in graph_1.edges and (node_i, node_j) in graph_2.edges:
                                attribute_1 = graph_1.edges[(node_a, node_b)]["geodesic_distance"]
                                attribute_2 = graph_2.edges[(node_i, node_j)]["geodesic_distance"]

                                # get the kernel value
                                kernel_args[1]["attribute_type"] = "geodesic"
                                value_kernel = kernel(attribute_1, attribute_2, kernel_args)
                                affinity_matrix[ node_a * graph_2.number_of_nodes() + node_i, node_b * graph_2.number_of_nodes() + node_j] \
                                    = value_kernel
                            
    return affinity_matrix


def get_matching(full_matching, graph_1_num, graph_2_num, nb_graphs):
    """
    Given the complete matching matrix, get the matching between graph_1 and graph_2
    """

    # check that graph_1 is a lower number than graph_2 otherwise inverse them
    if graph_1_num > graph_2_num:
        tmp = graph_2_num
        graph_2_num = graph_1_num
        graph_1_num = tmp
    
    nb_nodes = int(full_matching.shape[0] / nb_graphs)
    nodes_to_skip_1 = nb_nodes * graph_1_num
    nodes_to_skip_2 = nb_nodes * graph_2_num
    matching = full_matching[nodes_to_skip_1:nodes_to_skip_1 + nb_nodes,
                             nodes_to_skip_2:nodes_to_skip_2+nb_nodes]

    print("matching size :", matching.shape)
    return matching
    

def get_matching_affinity(affinity_matrix, matching):
    """
    Given a complete affinity matrix and a matching, return the affinity value of this matching
    """

    vectorized_matching = matching.flatten()
    matching_value = vectorized_matching @ affinity_matrix @ vectorized_matching
    print("matching value :", matching_value)
    return matching_value

    
def load_generate_full_affinity(path_to_folder, graph_nb_1, graph_nb_2, kernel_args):
    """ Generate the full affinity matrix and save it
        in a given repositery
    """

    print(path_to_folder, graph_nb_1, graph_nb_2)
    
    # get the two graphs
    graph_1 = nx.read_gpickle(os.path.join(path_to_folder, "modified_graphs", "graph_"+str(graph_nb_1)+".gpickle"))
    graph_2 = nx.read_gpickle(os.path.join(path_to_folder, "modified_graphs", "graph_"+str(graph_nb_2)+".gpickle"))

    # if the kernel is gaussian get the gamma value for the coordinate
    # and the geodesic distance
    if kernel_args[0] == "gaussian" and kernel_args[1]["gaussian_gamma"] == 0:
        gamma_coord, gamma_geodesic = compute_heuristic_gamma(graph_1, graph_2)
        print("gamma coord:",gamma_coord,"gamma geo:", gamma_geodesic)
        kernel_args[1]["gaussian_gamma_coord"] = gamma_coord
        kernel_args[1]["gaussian_gamma_geodesic"] = gamma_geodesic

    full_aff_matrix = full_affinity(graph_1, graph_2, kernel_args)
    return full_aff_matrix


def load_and_get_affinity_matching(path_to_folder, graph_nb_1, graph_nb_2, matching, kernel_args):
    """
    Load and generate the full affinity matrix to get the
    afinity value of the matching
    """

    # generate the affinity matrix
    affinity_matrix = load_generate_full_affinity(path_to_folder, graph_nb_1, graph_nb_2, kernel_args)

    # Get the value of the matching
    matching_value = get_matching_affinity(affinity_matrix, matching)

    return matching_value.item()


def generate_and_save_all_full_affinity_value(path_to_folder, kernel_args, nb_workers=4):
    """ Go through all folders and subfolders to load graphs and
        generate the correponding affinity and incidence matrices.
        This process is done using subprocesses to increase the
        computation time
    """

    # Initialise the list of arguments for the pool function
    list_arguments = []

    # get the full matching matrix
    path_to_matching = os.path.join(path_to_folder, "X_mALS")
    full_matching = sio.loadmat(path_to_matching)["X"]

    # get the number of graph
    nb_tot_graphs = len(os.listdir(os.path.join(path_to_folder, "modified_graphs")))

    # For each pair of graphs
    for i_graph in range(nb_tot_graphs):
        for j_graph in range(i_graph+1, nb_tot_graphs):

            # get the matching for these graphs
            sub_matching = get_matching(full_matching, i_graph, j_graph, nb_tot_graphs)
            list_arguments.append((path_to_folder, i_graph, j_graph, sub_matching, kernel_args))


    # launch the processes
    with Pool(processes=nb_workers) as pool:

        result_list = pool.starmap(load_and_get_affinity_matching, list_arguments)

    # put all the result in a dict (easier to understand and read)
    list_relation = []
    for i in range(nb_tot_graphs - 1):
        for j in range(i + 1, nb_tot_graphs):
            list_relation.append((i,j))

    dict_result = {}
    for position_in_result, (i,j) in enumerate(list_relation):

        if i not in dict_result:
            dict_result[i] = {}
        
        dict_result[i][j] = result_list[position_in_result]

    # save the dict
    pickle_out = open(os.path.join(path_to_folder,"matching_values.pickle"),"wb")
    pickle.dump(dict_result, pickle_out)
    pickle_out.close()
        


    
if __name__ == "__main__":

    # We parse the argument from command line
    parser = argparse.ArgumentParser(description="Generate the affinity and incidence matrices for preprocessed graphs of real data")
    parser.add_argument("path_to_folder", help="path where the folders contains the graphs")
    parser.add_argument("--nb_workers", help="number of processes to launch", default=1, type=int)
    parser.add_argument("--kernel_type", help="kernel type, only gaussian right now", default="gaussian")
    parser.add_argument("--gaussian_gamma", help="gamma value for the gaussian kernel", default=0, type=float)
    args = parser.parse_args()

    
    path_to_folder = args.path_to_folder
    nb_workers = args.nb_workers
    gaussian_gamma = args.gaussian_gamma
    kernel_type = args.kernel_type
    
    # We define the kernel arguments to be used
    kernel_args = (kernel_type, {"gaussian_gamma":gaussian_gamma})
    
    generate_and_save_all_full_affinity_value(path_to_folder,
                                                kernel_args,
                                                nb_workers=nb_workers)
 

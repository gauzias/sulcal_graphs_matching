import os
import numpy as np
import networkx as nx
import pickle


def list_to_dict(list_in):
    """
    converter used for pitsgraph to networkx conversion
    :param array:
    :return:
    """
    D = {}
    for i, l_i in enumerate(list_in):
        D[i] = l_i
    return D


def sphere_nearest_neighbor_interpolation(graph, sphere_mesh, coord_attribute='coord'):
    """
    For each node in the graph,
    find the closest vertex in the sphere mesh from the 'coord' attribute of each node
    :param graph:
    :param sphere_mesh:
    :return:
    """

    nodes_coords = graph_nodes_attribute(graph, coord_attribute)
    vertex_number1 = sphere_mesh.vertices.shape[0]

    #print('vert_template.shape', vert_template.shape[0])
    #print('vert_pits.shape', vert_pits.shape[0])
    nn = np.zeros(nodes_coords.shape[0], dtype=np.int64)
    for ind, v in enumerate(nodes_coords):
        #print(v)
        nn_tmp = np.argmin(np.sum(np.square(np.tile(v, (vertex_number1, 1)) - sphere_mesh.vertices), 1))
        nn[ind] = nn_tmp
    #print(nodes_coords.shape)
    #print(len(nn))
    #nx.set_node_attributes(graph, list_to_dict(nn), 'ico100_7_vertex_index_noreg')
    nx.set_node_attributes(graph, list_to_dict(nn), 'ico100_7_vertex_index') # Non Registered Vertex

    #return graph


def load_graphs_in_list(path_to_graphs_folder, suffix=".gpickle"):
    """
    Return a list of graph loaded from the path, ordered according to the filename on disk
    """
    g_files = []
    with os.scandir(path_to_graphs_folder) as files:
        for file in files:
            if file.name.endswith(suffix):
                g_files.append(file.name)

    g_files.sort()  # sort according to filenames

    list_graphs = [nx.read_gpickle(os.path.join(path_to_graphs_folder,graph)) for graph in g_files]

    return list_graphs


def load_labelled_graphs_in_list(path_to_graphs_folder, hemi='lh'):
    """
    Return a list of graph loaded from the path
    """
    files = os.listdir(path_to_graphs_folder).sort()
    files_to_load = list()
    for f in files:
        if '.gpickle' in f:
            if hemi in f:
                files_to_load.append(f)
    list_graphs = []
    for file_graph in files_to_load:
        path_graph = os.path.join(path_to_graphs_folder, file_graph)
        graph = nx.read_gpickle(path_graph)
        list_graphs.append(graph)

    return list_graphs


def graph_nodes_to_coords(graph, index_attribute, mesh):
    vert_indices = list(nx.get_node_attributes(graph, index_attribute).values())
    coords = np.array(mesh.vertices[vert_indices, :])
    return coords


def add_nodes_attribute(graph, list_attribute, attribute_name):
    """
    Given a graph, add to each node the corresponding attribute
    """

    attribute_dict = {}
    for node in graph.nodes:
        attribute_dict[node] = {attribute_name:list_attribute[node]}
    nx.set_node_attributes(graph, attribute_dict)


def graph_nodes_attribute(graph, attribute):
    """
    get the 'attribute' node attribute from 'graph' as a numpy array
    :param graph: networkx graph object
    :param attribute: string, node attribute to be extracted
    :return: a numpy array where i'th element corresponds to the i'th node in the graph
    if 'attribute' is not a valid node attribute in graph, then the returned array is empty
    """
    att = list(nx.get_node_attributes(graph, attribute).values())
    return np.array(att)


def graph_edges_attribute(graph, attribute):
    """
    get the 'attribute' edge attribute from 'graph' as a numpy array
    :param graph: networkx graph object
    :param attribute: string, node attribute to be extracted
    :return: a numpy array where i'th element corresponds to the i'th edge in the graph
    if 'attribute' is not a valid attribute in graph, then the returned array is empty
    """
    att = list(nx.get_edge_attributes(graph, attribute).values())
    return np.array(att)


def remove_dummy_nodes(graph):
    is_dummy = graph_nodes_attribute(graph, 'is_dummy')
    data_mask = np.ones_like(is_dummy)
    if True in is_dummy:
        graph.remove_nodes_from(np.where(np.array(is_dummy) == True)[0])
        inds_dummy = np.where(np.array(is_dummy)==True)[0]
        data_mask[inds_dummy] = 0
    return data_mask

# def remove_dummy_nodes(graph):
#     is_dummy = graph_nodes_attribute(graph, 'is_dummy')
#     data_mask = np.ones_like(is_dummy)
#     if True in is_dummy:
#         graph_copy = graph.copy()
#         inds_dummy = np.where(np.array(is_dummy)==True)[0]
#         graph_copy.remove_nodes_from(inds_dummy)
#         data_mask[inds_dummy] = 0
#         return graph_copy, data_mask
#     else:
#         return graph, data_mask

def compute_geodesic_distance_sphere(coord_a, coord_b, radius):
    '''
    Return the geodesic distance of two 3D vectors on a sphere
    '''
    return radius * np.arccos(np.clip(np.dot(coord_a, coord_b) / np.power(radius, 2), -1, 1))


def add_geodesic_distance_on_edges(graph):
    """
    Compute the geodesic distance represented by each edge
    and add it as attribute in the graph
    """

    # initialise the dict for atttributes on edges
    edges_attributes = {}

    # Fill the dictionnary with the geodesic_distance
    for edge in graph.edges:
        geodesic_distance = compute_geodesic_distance_sphere(graph.nodes[edge[0]]["sphere_3dcoords"],
                                                         graph.nodes[edge[1]]["sphere_3dcoords"],
                                                         radius=100)

        edges_attributes[edge] = {"geodesic_distance": geodesic_distance}

    nx.set_edge_attributes(graph, edges_attributes)


def add_id_on_edges(graph):
    """
    Add an Id information on edge (integer)
    """

    # initialise the dict for atttributes on edges
    edges_attributes = {}

    # Fill the dictionnary with the geodesic_distance
    for i, edge in enumerate(graph.edges):
        edges_attributes[edge] = {"id": i}

    nx.set_edge_attributes(graph, edges_attributes)


def add_dummy_nodes(graph, nb_node_to_reach):
    """
    Add a given number of dummy nodes to the graph
    """
    for _ in range(graph.number_of_nodes(), nb_node_to_reach):
        graph.add_node(graph.number_of_nodes(), is_dummy=True)


def transform_3dcoords_attribute_into_ndarray(graph):
    """
    Transform the node attribute sphere_3dcoord from a list to a ndarray
    """
    # initialise the dict for atttributes on edges
    nodes_attributes = {}

    # Fill the dictionnary with the nd_array attribute
    for node in graph.nodes:
        nodes_attributes[node] = {"sphere_3dcoords": np.array(graph.nodes[node]["sphere_3dcoords"])}

    nx.set_node_attributes(graph, nodes_attributes)


def preprocess_graph(graph):
    """
    preprocessing of graphs
    :param graph:
    :return:
    """

    # transform the 3d attributes into ndarray
    transform_3dcoords_attribute_into_ndarray(graph)

    # Compute the geodesic distance for each node and add the id information
    add_geodesic_distance_on_edges(graph)

    # add ID identifier on edges
    add_id_on_edges(graph)

    # add the 'is_dummy' attribute to nodes, that will be used when manipulating dummy nodes later
    nx.set_node_attributes(graph, values=False, name="is_dummy")





###################################################################
# main function coded by Nathan to preprocess all real data graphs
###################################################################
def read_modify_and_write_graphs(path_to_folder):
    """
    Read a list of graph in a folder, add dummy nodes where it's
    necessary in order to have an equal number of nodes between all graphs
    and finally write the modified graphs in a new folder with a
    dictionnary to allow the correspondences.
    """

    # Initialise correspondence dictionary
    correspondence_dict = {}

    # initialise list of graphs
    graph_list = []

    # load all the graphs one after the other
    for graph_i, graph_file in enumerate([file_name for file_name in os.listdir(path_to_folder) if
                                          not os.path.isdir(os.path.join(path_to_folder, file_name))]):
        # load this graph
        graph = nx.read_gpickle(os.path.join(path_to_folder, graph_file))
        preprocess_graph(graph)
        graph_list.append(graph)  # add it to the list of graph
        correspondence_dict[graph_i] = {"name": graph_file}  # add the information about the name.


    # add the number of nodes information to the dict and find the max number of nodes
    max_nb_nodes = 0
    for i, graph in enumerate(graph_list):
        correspondence_dict[i]["nb_nodes"] = graph.number_of_nodes()
        if graph.number_of_nodes() > max_nb_nodes:
            max_nb_nodes = graph.number_of_nodes()

    # add the dummy nodes
    for graph in graph_list:
        add_dummy_nodes(graph, max_nb_nodes)

    # Create the new folder for the graphs
    new_folder_path = os.path.join(path_to_folder, "modified_graphs")
    if not os.path.isdir(new_folder_path):
        os.mkdir(new_folder_path)

    # Save the graphs in the new folder
    for i, graph in enumerate(graph_list):
        graph_path = os.path.join(new_folder_path, "graph_" + str(i) + ".gpickle")
        nx.write_gpickle(graph, graph_path)

    # Save the correspondence_dict
    pickle_out = open(os.path.join(path_to_folder, "correspondence_dict.pickle"), "wb")
    pickle.dump(correspondence_dict, pickle_out)
    pickle_out.close()



import os
import sys
sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching/")
import argparse
import numpy as np
import networkx as nx
import slam.plot as splt
import slam.topology as stop
import slam.generate_parametric_surfaces as sgps
import trimesh
import os
import tools.graph_processing as gp
from sphere import *
from tqdm.auto import tqdm,trange
from scipy.stats import betabinom
import random



def generate_reference_graph(nb_vertices, radius):
    # Generate random sampling
    sphere_random_sampling = generate_sphere_random_sampling(vertex_number=nb_vertices, radius=radius)

    sphere_random_sampling = tri_from_hull(sphere_random_sampling)  # Computing convex hull (adding edges)

    adja = stop.adjacency_matrix(sphere_random_sampling)
    graph = nx.from_numpy_matrix(adja.todense())
    # Create dictionnary that will hold the attributes of each node
    node_attribute_dict = {}
    for node in graph.nodes():
        node_attribute_dict[node] = {"coord": np.array(sphere_random_sampling.vertices[node])}

    # add the node attributes to the graph
    nx.set_node_attributes(graph, node_attribute_dict)

    # We add a default weight on each edge of 1
    nx.set_edge_attributes(graph, 1.0, name="weight")

    # We add a geodesic distance between the two ends of an edge
    edge_attribute_dict = {}
    id_counter = 0  # useful for affinity matrix caculation
    for edge in graph.edges:
        # We calculate the geodesic distance
        end_a = graph.nodes()[edge[0]]["coord"]
        end_b = graph.nodes()[edge[1]]["coord"]
        geodesic_dist = gp.compute_geodesic_distance_sphere(end_a, end_b, radius)

        # add the information in the dictionnary
        edge_attribute_dict[edge] = {"geodesic_distance": geodesic_dist, "id": id_counter}
        id_counter += 1

    # add the edge attributes to the graph
    nx.set_edge_attributes(graph, edge_attribute_dict)

    return graph


def tri_from_hull(vertices):
    """
	compute faces from vertices using trimesh convex hull
	:param vertices: (n, 3) float
	:return:
	"""
    mesh = trimesh.Trimesh(vertices=vertices, process=False)
    return mesh.convex_hull


def edge_len_threshold(graph,thr): # Adds a percentage of edges 
    
    edge_to_add = random.sample(list(graph.edges),round(len(graph.edges)*thr))

    return edge_to_add


def generate_sphere_random_sampling(vertex_number=100, radius=1.0):
    """
	generate a sphere with random sampling
	:param vertex_number: number of vertices in the output spherical mesh
	:param radius: radius of the output sphere
	:return:
	"""
    coords = np.zeros((vertex_number, 3))
    for i in range(vertex_number):
        M = np.random.normal(size=(3, 3))
        Q, R = np.linalg.qr(M)
        coords[i, :] = Q[:, 0].transpose() * np.sign(R[0, 0])
    if radius != 1:
        coords = radius * coords
    return coords


def compute_beta(alpha, n, mean):
    return (1-mean/n) / (mean/n) * alpha


def compute_alpha(n, mean, variance):
    ratio = (1-mean/n) / (mean/n)
    alpha = ((1+ratio)**2 * variance - n**2 * ratio) / (n*ratio*(1+ratio) - variance* (1 + ratio)**3)
    return alpha


def generate_nb_outliers_and_nb_supress(nb_vertices):

    """

    Sample nb_outliers and nb_supress from a Normal dist
    following the std of real data

    """

    #mean_real_data = 40         # mean real data
    std_real_data = 3           # std real data


    mu = 12 # mu_A = mu_B = mu
    sigma = std_real_data
    n = 30

    alpha = compute_alpha(n , mu, sigma**2)  # corresponding alpha with respect to given mu and sigma
    beta = compute_beta(alpha, n, mu)       # corresponding beta

    nb_supress = betabinom.rvs(n, alpha, beta, size=1)[0]
    nb_outliers = betabinom.rvs(n, alpha, beta, size=1)[0]                 # Sample nb_outliers


    return int(nb_outliers),int(nb_supress)



def generate_noisy_graph(original_graph, nb_vertices, sigma_noise_nodes=1, sigma_noise_edges=1,
                         radius=100):
    # Perturbate the coordinates

    noisy_coord = []
    key = []
    value = []

    for index in range(nb_vertices):
        # Sampling from Von Mises - Fisher distribution
        original_coord = original_graph.nodes[index]["coord"]
        mean_original = original_coord / np.linalg.norm(original_coord)  # convert to mean unit vector.
        noisy_coordinate = Sphere().sample(1, distribution='vMF', mu=mean_original,
                                           kappa=sigma_noise_nodes).sample[0]

        noisy_coordinate = noisy_coordinate * np.linalg.norm(original_coord) # rescale to original size.
        # print(noisy_coordinate)
        noisy_coord.append(noisy_coordinate)


    nb_outliers, nb_supress = generate_nb_outliers_and_nb_supress(nb_vertices)  # Sample nb_outliers and nb_supress
    #nb_outliers = 0 # TEMPORARILY
    #nb_supress = 0


    noisy_coord_all = noisy_coord

    #Supress Non-Outlier nodes
    if nb_supress > 0:

        # print("nb_supress : ",nb_supress)

        supress_list = random.sample(range(len(noisy_coord)), nb_supress) # Indexes to remove 
        removed_coords = [noisy_coord[i] for i in range(len(noisy_coord)) if i in supress_list]
        #noisy_coord = [dummy_coords if i in supress_list else noisy_coord[i] for i in range(len(noisy_coord))]
        noisy_coord = [noisy_coord[i] for i in range(len(noisy_coord)) if i not in supress_list]


    # Add Outliers
    sphere_random_sampling = []
    if nb_outliers > 0:

        #print("nb_outliers: ", nb_outliers)

        sphere_random_sampling = generate_sphere_random_sampling(vertex_number=nb_outliers, radius=radius)
        # merge pertubated and outlier coordinates to add edges 
        all_coord = noisy_coord + list(sphere_random_sampling)
    else:
        all_coord = noisy_coord


    noisy_graph = nx.Graph()

    compute_noisy_edges = tri_from_hull(all_coord)  # take all peturbated coord and comp conv hull.
    adja = stop.adjacency_matrix(compute_noisy_edges)  # compute the new adjacency mat.

    noisy_graph = nx.from_numpy_matrix(adja.todense())

    node_attribute_dict = {}
    for node in noisy_graph.nodes():
        node_attribute_dict[node] = {"coord": np.array(compute_noisy_edges.vertices[node]),'is_dummy':False,'is_outlier':False}



    nx.set_node_attributes(noisy_graph, node_attribute_dict)

    nx.set_edge_attributes(noisy_graph, 1.0, name="weight")

    edge_attribute_dict = {}
    id_counter = 0  # useful for affinity matrix caculation
    for edge in noisy_graph.edges:
        # We calculate the geodesic distance
        end_a = noisy_graph.nodes()[edge[0]]["coord"]
        end_b = noisy_graph.nodes()[edge[1]]["coord"]
        geodesic_dist = gp.compute_geodesic_distance_sphere(end_a, end_b, radius)

        # add the information in the dictionnary
        edge_attribute_dict[edge] = {"geodesic_distance": geodesic_dist, "id": id_counter}
        id_counter += 1

    # add the edge attributes to the graph
    nx.set_edge_attributes(noisy_graph, edge_attribute_dict)

    # Extracting the ground-truth correspondence

    ground_truth_permutation = []
    counter = 0
    check = False


    
    
    for i in range(len(noisy_graph.nodes)): 
        for j in range(len(noisy_coord_all)):  # upto the indexes of outliers
            if np.linalg.norm(noisy_coord_all[j] - noisy_graph.nodes[i]['coord']) == 0.:
                ground_truth_permutation.append(j)
                continue
                
            elif j == len(noisy_coord_all) - 1.:
                 for outlier in sphere_random_sampling:
                        if np.linalg.norm(outlier - noisy_graph.nodes[i]['coord']) == 0.:
                            noisy_graph.nodes[i]['is_outlier'] = True

                            ground_truth_permutation.append(-1)
        



    # for outlier in sphere_random_sampling:
    #     for i in range(len(noisy_graph.nodes)):

    #         if np.linalg.norm(outlier - noisy_graph.nodes[i]['coord']) == 0.:

    #             if i<nb_vertices:
    #                 value.append(i)



    # if nb_outliers > 0 and len(key)!=0:
    #     index = 0
    #     for j in range(len(ground_truth_permutation)):
    #         if ground_truth_permutation[j] == key[index]:
    #             ground_truth_permutation[j] = value[index]
    #             index+=1
    #             if index == len(key):
    #                 break


    #     key = key + value
    #     value = value + key

    #     mapping = dict(zip(key,value))
    #     #print("mapping :",mapping)
    #     #print("number of nodes in graphs: ", len(noisy_graph.nodes))
    #     noisy_graph = nx.relabel_nodes(noisy_graph, mapping)


    # Remove 10% of random edges
    edge_to_remove = edge_len_threshold(noisy_graph, 0.10)
    noisy_graph.remove_edges_from(edge_to_remove)

    noisy_graph.remove_edges_from(nx.selfloop_edges(noisy_graph))



    return ground_truth_permutation, noisy_graph


def get_nearest_neighbors(original_coordinates, list_neighbors, radius, nb_to_take=10):
    ''' Return the nb_to_take nearest neighbors (in term of geodesic distance) given a set
		of original coordinates and a list of tuples where the first term is the label 
		of the node and the second the associated coordinates
	'''

    # We create the list of distances and sort it
    distances = [(i, gp.compute_geodesic_distance_sphere(original_coordinates, current_coordinates, radius)) for
                 i, current_coordinates in list_neighbors]
    distances.sort(key=lambda x: x[1])

    return distances[:nb_to_take]


def add_integer_id_to_edges(graph):
    """ Given a graph, add an attribute "id" to each edge that is a unique integer id"""

    dict_attributes = {}
    id_counter = 0
    for edge in graph.edges:
        dict_attributes[edge] = {"id": id_counter}
        id_counter += 1
    nx.set_edge_attributes(graph, dict_attributes)


def mean_edge_len(G):
    all_geo = [z['geodesic_distance'] for x, y, z in list(G.edges.data())]
    #mean_geo = np.array(all_geo).mean()
    # std = np.std(all_geo)

    return all_geo


def get_in_between_perm_matrix(perm_mat_1, perm_mat_2):
    """
    Given two permutation from noisy graphs to a reference graph,
    Return the permutation matrix to go from one graph to the other
    """
    result_perm = {}
    for i in range(len(perm_mat_1)):
        if perm_mat_1[i] == -1:
                continue
        for j in range(len(perm_mat_2)):
            if perm_mat_2[j] == -1:
                continue
            
            if perm_mat_1[i] == perm_mat_2[j]:
                result_perm[i] = j

    return result_perm


# def get_in_between_perm_matrix_old(perm_mat_1, perm_mat_2):
#     """
#     Given two permutation from noisy graphs to a reference graph,
#     Return the permutation matrix to go from one graph to the other
#     """
#     result_perm = np.zeros((perm_mat_1.shape[0],), dtype=int)

#     for node_reference, node_noisy_1 in enumerate(perm_mat_1):
#         # get the corresponding node in the second graph
#         node_noisy_2 = perm_mat_2[node_reference]

#         # Fill the result
#         result_perm[node_noisy_1] = node_noisy_2

#     return result_perm





def generate_graph_family(nb_sample_graphs, nb_graphs, nb_vertices, radius, nb_outliers, ref_graph, noise_node=1, noise_edge=1):
    """
	Generate n noisy graphs from a reference graph alongside the 
	ground truth permutation matrices.
	"""
    # Generate the reference graph
    reference_graph = ref_graph

    # Initialise the list of noisy_graphs
    list_noisy_graphs = []
    list_ground_truth = []

    # We generate the n noisy graphs
    print("Generating graphs..")

    #for c_graph in tqdm(range(nb_sample_graphs)):
    count_graph =  0
    while count_graph < nb_graphs:


        ground_truth, noisy_graph = generate_noisy_graph(reference_graph, nb_vertices, noise_node, noise_edge)

        if nx.is_connected(noisy_graph) == False:
            print("Found disconnected components..!!")
            print("Regenerating noisy graph..!!")
            continue

        if nx.is_connected(noisy_graph) == True:
            print(count_graph,end='\r')
            count_graph += 1

        # Add id to edge
        add_integer_id_to_edges(noisy_graph)

        # Save the graph
        list_noisy_graphs.append(noisy_graph)

        # Save all ground-truth for later selecting the selected graphs
        list_ground_truth.append(ground_truth)



    # min_geo = []
    # selected_graphs = []
    # selected_ground_truth = []

    # for graphs,gt in zip(list_noisy_graphs,list_ground_truth):
    #     z = mean_edge_len(graphs)
    
    #     if min(z) > 7.0:
    #         selected_graphs.append(graphs) # select the noisy graph.
    #         selected_ground_truth.append(gt) # and its corresponding ground-truth.
    #         min_geo.append(min(z))


    # sorted_zipped_lists = zip(min_geo, selected_graphs, selected_ground_truth)
    # sorted_zipped_lists = sorted(sorted_zipped_lists,reverse = True)

    # sorted_graphs = []
    # sorted_ground_truth = []

    # for l,m,n in sorted_zipped_lists:
    #     sorted_graphs.append(m)
    #     sorted_ground_truth.append(n)

    sorted_graphs = list_noisy_graphs
    sorted_ground_truth = list_ground_truth


    #print("Verifying len of sorted_graphs,sorted_ground_truth,min_geo(should be equal):",len(sorted_graphs),len(sorted_ground_truth),len(min_geo))
    print("Verifying len of num_graphs,num_ground_truth,min_geo:",len(sorted_graphs),len(sorted_ground_truth))
 

    # # Initialise permutation matrices to reference graph
    # ground_truth_perm_to_ref = np.zeros((nb_graphs, nb_vertices), dtype=int)
    # ground_truth_perm = np.zeros((nb_graphs, nb_graphs, nb_vertices), dtype=int)



    # Save the ground_truth permutation
    # count = 0
    # for ground_truth in sorted_ground_truth[:nb_graphs]: # Select the nb_graphs with largest min-geo distance
    #     ground_truth_perm_to_ref[count, :len(ground_truth)] = ground_truth
    #     count +=1 


    ground_truth_perm_to_ref = sorted_ground_truth



    # We generate the ground_truth permutation between graphs
    print("Groundtruth Labeling..")

    ground_truth_perm = {}

    for i_graph in tqdm(range(nb_graphs)):

        for j_graph in range(nb_graphs):


            ##ground_truth_perm[i_graph, j_graph, :]=get_in_between_perm_matrix(ground_truth_perm_to_ref[i_graph, :], ground_truth_perm_to_ref[j_graph, :])

           ground_truth_perm[str(i_graph)+','+str(j_graph)] = get_in_between_perm_matrix(ground_truth_perm_to_ref[i_graph], ground_truth_perm_to_ref[j_graph])


    return sorted_graphs[:nb_graphs] , ground_truth_perm,ground_truth_perm_to_ref




def generate_n_graph_family_and_save(path_to_write, nb_runs, nb_ref_graph, nb_sample_graphs,nb_graphs, nb_vertices,
                                     radius, list_noise, save_reference=0):
    ''' Generate n family of graphs for each couple (noise, outliers). The graphs are saved
		in a folder structure at the point path_to_write
	'''

    # check if the path given is a folder otherwise create one
    if not os.path.isdir(path_to_write):
        os.mkdir(path_to_write)

    # generate n families of graphs
    for i_graph in range(nb_runs):

        # Select the ref graph with highest mean geo distance
        print("Generating reference_graph..")
        for i in tqdm(range(nb_ref_graph)):
            reference_graph = generate_reference_graph(nb_vertices, radius)
            all_geo = mean_edge_len(reference_graph)

            if i == 0:
                min_geo = min(all_geo)

            else:

                if min(all_geo) > min_geo:
                    min_geo = min(all_geo)
                    reference_graph_max = reference_graph

                else:
                    pass

        if save_reference:
            print("Selected reference graph with min_geo: ",min_geo)
            trial_path = os.path.join(path_to_write, str(i_graph))  # for each trial
            if not os.path.isdir(trial_path):
                os.mkdir(trial_path)
            nx.write_gpickle(reference_graph_max, os.path.join(trial_path, "reference_" + str(i_graph) + ".gpickle"))

        for noise in list_noise:
            #for outliers in list_outliers:

            folder_name = "noise_" + str(noise) + ",outliers_varied"  #+ str(max_outliers)
            path_parameters_folder = os.path.join(trial_path, folder_name)

            if not os.path.isdir(path_parameters_folder):
                os.mkdir(path_parameters_folder)
                os.mkdir(os.path.join(path_parameters_folder, "graphs"))

            list_graphs,ground_truth_perm,ground_truth_perm_to_ref  = generate_graph_family(nb_sample_graphs= nb_sample_graphs,nb_graphs=nb_graphs,
                                                                   nb_vertices=nb_vertices,
                                                                   radius=radius,
                                                                   ref_graph=reference_graph_max,
                                                                   noise_node=noise,
                                                                   noise_edge=noise)


            for i_family, graph_family in enumerate(list_graphs):

                sorted_graph = nx.Graph()
                sorted_graph.add_nodes_from(sorted(graph_family.nodes(data=True)))  # Sort the nodes of the graph by key
                sorted_graph.add_edges_from(graph_family.edges(data=True))

                print("Length of noisy graph: ",len(sorted_graph.nodes))


                nx.write_gpickle(sorted_graph, os.path.join(path_parameters_folder, "graphs","graph_{:05d}".format(i_family) + ".gpickle"))

            # np.save(os.path.join(path_parameters_folder, "ground_truth"), ground_truth_perm)

            nx.write_gpickle(ground_truth_perm_to_ref, path_parameters_folder+ "/permutation_to_ref_graph.gpickle")

            nx.write_gpickle(ground_truth_perm, path_parameters_folder+ "/ground_truth.gpickle")




if __name__ == '__main__':

    path_to_write = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/simu_graph/Small_set/'

    # Change only the following parameters according to need: nb_graphs, nb_vertices

    nb_runs = 1
    nb_sample_graphs = 10 # of graphs to generate before selecting the NN graphs with highest geodesic distance.
    nb_graphs = 137 # 137 # nb of graphs to generate
    nb_vertices = 88 # 88 as per real data mean (OASIS)  #72 based on Kaltenmark, MEDIA, 2020 // 88 based on the avg number of nodes in the real data.
    #max_outliers = 20
    #step_outliers = 10
    save_reference = 1
    nb_ref_graph = 10000
    radius = 100




    #list_noise = np.arange(min_noise, max_noise, step_noise)
    list_noise = np.array([100, 200, 400, 1000])

    # call the generation procedure 
    generate_n_graph_family_and_save(path_to_write=path_to_write,
                                     nb_runs=nb_runs,
                                     nb_ref_graph=nb_ref_graph,
                                     nb_sample_graphs=nb_sample_graphs,
                                     nb_graphs = nb_graphs,
                                     nb_vertices=nb_vertices,
                                     radius=radius,
                                     list_noise=list_noise,
                                     save_reference=save_reference)


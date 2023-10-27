import tools.graph_processing as gp
import numpy as np
import networkx as nx
import pickle as p
from visbrain.objects import SourceObj, ColorbarObj


def insert_at(arr, output_size, indices):
    # assert len(output_size) == len(indices) == len(arr.shape)
    result = np.zeros(output_size, dtype=np.uint16)
    existing_indices = [np.setdiff1d(np.arange(axis_size), axis_indices, assume_unique=True)
                        for axis_size, axis_indices in zip(output_size, indices)]
    result[np.ix_(*existing_indices)] = arr
    return result


def get_assignment_from_labelling(list_graphs, labelling_attribute_name, excluded_labels=None):
    """
    compute the assignment matrix based on the labeling stored in the nodes of the graphs in list_graphs,
    as the node attribute labelling_attribute_name
    :param list_graphs: list of graphs to work on
    :param labelling_attribute_name: node attribute used to store the labeling for which we compute the assignment
    matrix
    :return: assign_mat the assignment matrix and unique_labels the set of labels found across all graphs and ordered
    according to the rows of the computed assign_mat
    """
    all_graphs_labels = get_labelling_from_attribute(list_graphs, labelling_attribute_name)
    list_all_graphs_labels = concatenate_labels(all_graphs_labels)
    unique_labels = list(set(list_all_graphs_labels))
    if excluded_labels is not None:
        print('excluded labels ', excluded_labels)
        for ex_lab in excluded_labels:
            unique_labels.remove(ex_lab)
    unique_labels.sort()
    tot_nb_nodes = len(list_all_graphs_labels)
    print('total number of nodes across all graphs', tot_nb_nodes)
    print('number of different labels stored in graphs', len(unique_labels))
    print('labels stored in graphs', unique_labels)
    # relabelling to get continuous labels that will correspond to row of the assignment matrix
    row_index_list_all_graphs_labels = list()
    for i in list_all_graphs_labels:
        if i in unique_labels:  # handle excluded labels: these are not relabeled
            idx = unique_labels.index(i)  # take the index of i in the set of labels
            row_index_list_all_graphs_labels.append(idx)  # making the relabeled list
    print('row index corresponding to labels', set(row_index_list_all_graphs_labels))

    assign_semimat = np.zeros((tot_nb_nodes, len(unique_labels)), dtype=np.uint16)
    if excluded_labels is not None:
        for ind_node, label in zip(range(tot_nb_nodes), row_index_list_all_graphs_labels):
            if label not in excluded_labels:
                assign_semimat[ind_node, label] = 1
    else:
        for ind_node, label in zip(range(tot_nb_nodes), row_index_list_all_graphs_labels):
                assign_semimat[ind_node, label] = 1

    X = assign_semimat @ assign_semimat.T

    sizes_dummy = [nx.number_of_nodes(g) for g in list_graphs]
    dummy_mask = [list(nx.get_node_attributes(graph, 'is_dummy').values()) for graph in list_graphs]
    dummy_mask = sum(dummy_mask, [])
    dummy_indexes = [i for i in range(len(dummy_mask)) if dummy_mask[i] == True]

    X_w_dummy = insert_at(X, (sum(sizes_dummy), sum(sizes_dummy)),
                               (dummy_indexes, dummy_indexes))

    return X, X_w_dummy, unique_labels


def concatenate_labels(all_labels):
    cat_labels = list()
    for l in all_labels:
        cat_labels.extend(l)
    return cat_labels


def nb_labelled_nodes_per_label(u_labs, all_labels):
    u_l_count = list()
    for u_l in u_labs:
        subj_u = list()
        for subj_labs in all_labels:
            subj_u.append(np.sum(subj_labs == u_l))
        #print(u_l)
        #print(subj_u)
        u_l_count.append(subj_u)
    return np.array(u_l_count)


def label_nodes_according_to_coord(graph_no_dummy, template_mesh, coord_dim=1):
    nodes_coords = gp.graph_nodes_to_coords(graph_no_dummy, 'ico100_7_vertex_index', template_mesh)

    one_nodes_coords = nodes_coords[:, coord_dim]
    one_nodes_coords_scaled = (one_nodes_coords - np.min(one_nodes_coords))/(np.max(one_nodes_coords)-np.min(one_nodes_coords))

    #one_nodes_coords_scaled = np.random.rand(len(nodes_coords))
    # initialise the dict for atttributes
    nodes_attributes = {}
    # Fill the dictionnary with the nd_array attribute
    for ind, node in enumerate(graph_no_dummy.nodes):
        nodes_attributes[node] = {"label_color": one_nodes_coords_scaled[ind]}

    nx.set_node_attributes(graph_no_dummy, nodes_attributes)
    return one_nodes_coords_scaled


def get_labelling_from_attribute(list_graphs, labelling_attribute_name):
    all_labels = list()
    for g in list_graphs:
        labels = list(nx.get_node_attributes(g, labelling_attribute_name).values())
        all_labels.append(labels)

    return all_labels


def get_labelling_from_assignment(list_graphs, matching_matrix, largest_ind, mesh, labelling_attribute_name, default_label_value):
    nb_graphs = len(list_graphs)
    g_l = list_graphs[largest_ind]
    color_label_ordered = label_nodes_according_to_coord(g_l, mesh, coord_dim=0)
    r_perm = p.load(open("../data/r_perm_22.gpickle","rb"))
    color_label = color_label_ordered[r_perm]
    gp.add_nodes_attribute(g_l, color_label, labelling_attribute_name)
    default_value = default_label_value
    nb_nodes = len(g_l.nodes)
    row_scope = range(largest_ind * nb_nodes, (largest_ind + 1) * nb_nodes)

    #nb_unmatched = 0
    transfered_labels_all_graphs = list()
    for i in range(nb_graphs):
        g = list_graphs[i]
        if len(g.nodes) == nb_nodes:
            col_scope = range(i * nb_nodes, (i + 1) * nb_nodes)
            perm_X = np.array(matching_matrix[np.ix_(row_scope, col_scope)], dtype=int) #Iterate through each Perm Matrix X fixing the largest graph
            transfered_labels = np.ones(nb_nodes)*default_value
            for node_indx,ind in enumerate(row_scope):
                match_index = np.where(perm_X[node_indx,:]==1)[0]

                if len(match_index)>0:
                    transfered_labels[match_index[0]] = color_label[node_indx]
            #nb_unmatched += np.sum(transfered_labels==default_value)
            transfered_labels_all_graphs.append(transfered_labels)
            gp.add_nodes_attribute(g, transfered_labels, labelling_attribute_name)
        else:
            raise Exception("All graphs should have {:d} nodes, incl. dummy nodes".format(nb_nodes))
    return transfered_labels_all_graphs


def get_labelling_from_assignment_hippi(list_graphs, matching_matrix, largest_ind, mesh, labelling_attribute_name):
    nb_graphs = len(list_graphs)
    g_l = list_graphs[largest_ind]
    color_label_ordered = label_nodes_according_to_coord(g_l, mesh, coord_dim=1)
    r_perm = p.load(open("/mnt/data/work/python_sandBox/Graph_matching/data/r_perm.gpickle","rb"))
    color_label = color_label_ordered[r_perm]
    gp.add_nodes_attribute(g_l, color_label, labelling_attribute_name)
    default_value = -0.1#0.05
    nb_nodes = len(g_l.nodes)

    for j in range(len(list_graphs)):

        grph = list_graphs[j]
        nodes_to_remove = gp.remove_dummy_nodes(grph)
        nodes_to_remove = np.where(np.array(nodes_to_remove)==False)

        grph.remove_nodes_from(list(nodes_to_remove[0]))
        nb_nodes = len(grph.nodes)
        row_scope = range(j * nb_nodes, (j + 1) * nb_nodes)

        print(len(grph.nodes))

        if len(grph.nodes)==101:
            break


    print(matching_matrix.shape)
    last_index = 0

    nb_unmatched = 0
    for i in range(nb_graphs):

        #g = list_graphs[i]
        #g=p.load(open("../data/OASIS_full_batch/modified_graphs/graph_"+str(i)+".gpickle","rb"))
        g = list_graphs[i]
        nodes_to_remove = gp.remove_dummy_nodes(g)
        nodes_to_remove = np.where(np.array(nodes_to_remove)==False)
        g.remove_nodes_from(list(nodes_to_remove[0]))
        nb_nodes = len(g.nodes)

        print(len(g.nodes))

        if i == 0:
            col_scope = range(i * nb_nodes, (i + 1) * nb_nodes)
            prev_nb_nodes = nb_nodes
            perm_X = np.array(matching_matrix[np.ix_(row_scope, col_scope)], dtype=int) #Iterate through each Perm Matrix X fixing the largest graph
            transfered_labels = np.ones(nb_nodes)*default_value
            last_index+=nb_nodes
        else:
            col_scope = range(last_index, last_index+nb_nodes)
            last_index += nb_nodes
            perm_X = np.array(matching_matrix[np.ix_(row_scope, col_scope)], dtype=int) #Iterate through each Perm Matrix X fixing the largest graph
            transfered_labels = np.ones(nb_nodes)*default_value


        print(col_scope)

        #nb_nodes = len(g.nodes)
        #col_scope = range(i * nb_nodes, (i + 1) * nb_nodes)

        for node_indx,ind in enumerate(row_scope):
            match_index = np.where(perm_X[node_indx,:]==1)[0]

            if len(match_index)>0:
                transfered_labels[match_index[0]] = color_label[node_indx]
        gp.add_nodes_attribute(g, transfered_labels, labelling_attribute_name)
    return transfered_labels


def create_clusters_lists(list_graphs, label_attribute="label_dbscan"):
    """
    Given a list of graphs, return a list of list that represents the clusters.
    Each inside list represent one cluster and each elemnt of the cluster is
    a tuple (graph_number, node_in_the_graph).
    """

    result_dict = {}

    for i_graph, graph in enumerate(list_graphs):
        for node in graph.nodes:
            if not graph.nodes[node]["is_dummy"]:
                label_cluster = graph.nodes[node][label_attribute]
                if label_cluster in result_dict:
                    result_dict[label_cluster].append((i_graph, node))
                else:
                    result_dict[label_cluster] = [(i_graph, node)]

    # We make sure that every clusters have more than one element
    #{i: result_dict[i] for i in result_dict if len(result_dict[i]) > 1}
    return result_dict


def get_centroid_clusters(list_graphs, clusters_dict, coords_attribute="sphere_3dcoords"):
    """
    Return a dictionary which gives for each cluster the belonging point
    which is the closest to the centroid
    """

    result_dict = {}

    for cluster_key in clusters_dict:

        # initialise the matrix which holds the position of all the point in the cluster
        position_mat = np.zeros((len(clusters_dict[cluster_key]), 3))

        # fill the matrix
        for elem_i, (graph_num, node) in enumerate(clusters_dict[cluster_key]):
            graph = list_graphs[graph_num]
            position_mat[elem_i, :] = graph.nodes[node][coords_attribute]

        # get the centroid
        centroid = position_mat.mean(0)

        # get the closest point to the centroid
        min_distance = -1
        for graph_num, node in clusters_dict[cluster_key]:

            graph = list_graphs[graph_num]
            position_node = graph.nodes[node][coords_attribute]
            distance_to_centroid = np.linalg.norm(centroid - position_node)

            if distance_to_centroid < min_distance or min_distance == -1:
                min_distance = distance_to_centroid
                centroid_node = (graph_num, node)

        result_dict[cluster_key] = centroid_node

    return result_dict


def get_all_silhouette_value(list_graphs, cluster_dict):
    """
    Return a dict with all the silhouette value that can be calculated for each cluster
    https://en.wikipedia.org/wiki/Silhouette_(clustering)
    """
    result_dict = {}

    for cluster_key in cluster_dict:
        print('working on cluster:: ', cluster_key)
        nb_elem = len(cluster_dict[cluster_key])
        if nb_elem<2:
            silhouette = 0
            if cluster_key in result_dict:
                result_dict[cluster_key].append(silhouette)
            else:
                result_dict[cluster_key] = [silhouette]

        else:
            for main_counter in range(nb_elem):
                graph_main, node_main = cluster_dict[cluster_key][main_counter]
                vector_1 = list_graphs[graph_main].nodes[node_main]["sphere_3dcoords"]
                # We compute the distance across points within the same cluster
                a_list = []
                for intra_cluster_counter in range(nb_elem):

                    if main_counter != intra_cluster_counter:
                        graph_inter, node_inter = cluster_dict[cluster_key][intra_cluster_counter]
                        vector_2 = list_graphs[graph_inter].nodes[node_inter]["sphere_3dcoords"]
                        distance = np.linalg.norm(vector_1 - vector_2)
                        a_list.append(distance)

                a = np.sum(a_list) / (nb_elem - 1)

                # We compute the average distance with points from other clusters
                b_list = []
                for cluster_inter_key in cluster_dict:
                    if cluster_inter_key != cluster_key:
                        distance_list = []
                        for intra_cluster_counter in range(len(cluster_dict[cluster_inter_key])):
                            graph_intra, node_intra = cluster_dict[cluster_inter_key][intra_cluster_counter]
                            vector_2 = list_graphs[graph_intra].nodes[node_intra]["sphere_3dcoords"]
                            distance = np.linalg.norm(vector_1 - vector_2)
                            distance_list.append(distance)
                        b_list.append(np.mean(distance_list))
                b = np.min(b_list)

                silhouette = (b - a) / max(b, a)

                if cluster_key in result_dict:
                    result_dict[cluster_key].append(silhouette)
                else:
                    result_dict[cluster_key] = [silhouette]

    return result_dict


def get_silhouette_per_cluster(silhouette_dict):
    nb_clusters = len(silhouette_dict)
    silhouette_data = np.zeros(nb_clusters)
    cluster_nb_nodes = np.zeros(nb_clusters)

    # Get the data
    for cluster_i, cluster_key in enumerate(silhouette_dict):
        silhouette_data[cluster_i] = np.mean(silhouette_dict[cluster_key])
        cluster_nb_nodes[cluster_i] = len(silhouette_dict[cluster_key])
    return silhouette_data, cluster_nb_nodes


def get_silhouette_per_graph(cluster_dict, silhouette_dict, graph_ind, graph_nb_nodes):
    nb_clusters = len(silhouette_dict)
    nodes_silhouette = np.zeros(graph_nb_nodes)
    # Get the data
    for cluster_i, cluster_key in enumerate(silhouette_dict):
        silhouette_data = silhouette_dict[cluster_key]
        cluster_content = cluster_dict[cluster_key]
        for clus_ind, clus in enumerate(cluster_content):
            if clus[0] == graph_ind:
                nodes_silhouette[clus[1]] = silhouette_data[clus_ind]
    return nodes_silhouette


def get_consistency_per_cluster(clusters_dict, nodeCstPerGraph):
    nb_clusters = len(clusters_dict)
    cluster_cst = np.zeros(nb_clusters)

    for cluster_i, cluster_key in enumerate(clusters_dict):

        # initialise the matrix which holds the position of all the point in the cluster
        cluster_nodes_cst = np.zeros((len(clusters_dict[cluster_key]),))

        # fill the matrix
        for elem_i, (graph_num, node) in enumerate(clusters_dict[cluster_key]):
            cluster_nodes_cst[elem_i] = nodeCstPerGraph[node, graph_num]

        cluster_cst[cluster_i] = np.mean(cluster_nodes_cst)

    return cluster_cst


    nb_clusters = len(silhouette_dict)
    silhouette_data = np.zeros(nb_clusters)
    cluster_nb_nodes = np.zeros(nb_clusters)

    # Get the data
    for cluster_i, cluster_key in enumerate(silhouette_dict):
        silhouette_data[cluster_i] = np.mean(silhouette_dict[cluster_key])
        cluster_nb_nodes[cluster_i] = len(silhouette_dict[cluster_key])
    return silhouette_data, cluster_nb_nodes


def get_centroids_coords(centroid_dict, list_graphs, mesh, attribute_vertex_index="ico100_7_vertex_index"):

    nb_clusters = len(list(centroid_dict.keys()))
    centroids_3Dpos = np.zeros((nb_clusters, 3))

    # Get the data
    for cluster_i, cluster_key in enumerate(centroid_dict):
        graph_num, node = centroid_dict[cluster_key]
        graph = list_graphs[graph_num]

        vertex = graph.nodes[node][attribute_vertex_index]
        vertex_pos = mesh.vertices[vertex, :]
        # print(vertex_pos)
        # print(silhouette_3Dpos.shape)
        centroids_3Dpos[cluster_i, :] = vertex_pos
    return centroids_3Dpos


def compute_node_consistency(bulk_matrix, nb_graphs, nb_nodes):
    nodeCstPerGraph = np.zeros((nb_nodes, nb_graphs))
    for graph_ref_num in range(nb_graphs):
        print('graph_ref_num=', graph_ref_num)
        #rscope = (graph_ref_num - 1) * nb_nodes + 1:graph_ref_num * nb_nodes
        rscope = range(graph_ref_num * nb_nodes, (graph_ref_num + 1) * nb_nodes)
        for i in range(nb_graphs-1):
            #x_k_i = matching_matrix[, ]
            iscope = range(i * nb_nodes, (i+1)*nb_nodes)
            Xri = np.array(bulk_matrix[np.ix_(rscope, iscope)], dtype=int)
            for j in range(i+1,nb_graphs):
                jscope = range(j * nb_nodes, (j + 1) * nb_nodes)
                Xij = np.array(bulk_matrix[np.ix_(iscope, jscope)], dtype=int)
                Xrj = np.array(bulk_matrix[np.ix_(rscope, jscope)], dtype=int)
                Xrij = np.matmul(Xri, Xij)
                val = np.sum(np.abs(Xrij-Xrj), 1)
                nodeCstPerGraph[:, graph_ref_num] += val/2
                # if np.max(val)>10:
                #     print(val)
                #     toto


    # normalize the summation value
    nodeCstPerGraph = 1 - nodeCstPerGraph/(nb_graphs*(nb_graphs-1)/2)
    # clamp
    nodeCstPerGraph[nodeCstPerGraph < 0] = 0
    # sort
    # [~,IX] = np.sort(nodeCstPerGraph,1,'descend')
    # nodeCstPerGraph2 = np.zeros(nb_nodes,nb_graphs)
    # for ref in range(nb_graphs):
    #     nodeCstPerGraph2(IX(1:inCnt,ref),ref) = 1
    # nodeCstPerGraph = nodeCstPerGraph2
    return nodeCstPerGraph
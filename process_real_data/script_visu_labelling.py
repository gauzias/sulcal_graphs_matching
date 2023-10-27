import sys
#sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
import os
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import tools.clusters_analysis as gca
import numpy as np
import networkx as nx
import scipy.io as sco
import pickle as p
import copy


# def label_nodes_according_to_coord(graph_no_dummy, template_mesh, coord_dim=1):
#     nodes_coords = gp.graph_nodes_to_coords(graph_no_dummy, 'ico100_7_vertex_index', template_mesh)
#
#     one_nodes_coords = nodes_coords[:, coord_dim]
#     one_nodes_coords_scaled = (one_nodes_coords - np.min(one_nodes_coords))/(np.max(one_nodes_coords)-np.min(one_nodes_coords))
#
#     #one_nodes_coords_scaled = np.random.rand(len(nodes_coords))
#
#
#
#     # initialise the dict for atttributes
#     nodes_attributes = {}
#     # Fill the dictionnary with the nd_array attribute
#     for ind, node in enumerate(graph_no_dummy.nodes):
#         nodes_attributes[node] = {"label_color": one_nodes_coords_scaled[ind]}
#
#     nx.set_node_attributes(graph_no_dummy, nodes_attributes)
#     return one_nodes_coords_scaled

if __name__ == "__main__":
    #template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/ico100_7.gii'
    template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '../data/OASIS_full_batch/modified_graphs'
    path_to_silhouette = '/mnt/data/work/python_sandBox/Graph_matching/data/OASIS_full_batch'
    path_to_mALS = "../data/OASIS_full_batch/X_mALS.mat"
    path_to_mSync = "../data/OASIS_full_batch/X_mSync.mat"
    path_to_CAO = "../data/OASIS_full_batch/X_cao_cst_o.mat"
    path_to_kerGM = "../data/OASIS_full_batch/X_pairwise_kergm.mat"
    #path_to_match_mat = "/home/rohit/PhD_Work/GM_my_version/RESULT_FRIOUL_HIPPI/Hippi_res_real_mat.npy"

    list_graphs_cp = gp.load_graphs_in_list(path_to_graphs)
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    algorithms = []

    X_mALS = sco.loadmat(path_to_mALS)['X']
    X_mSync = sco.loadmat(path_to_mSync)['X']
    X_CAO = sco.loadmat(path_to_CAO)['X']
    x_kerGM = sco.loadmat(path_to_kerGM)["full_assignment_mat"]

    matching_matrix = X_mALS#x_kerGM#,X_mSync]#,X_CAO]
    label_attribute = 'labelling_mALS'
    #X = np.load(path_to_match_mat)

    nb_graphs = 134

    mesh = sio.load_mesh(template_mesh)

    largest_ind=24
    g_l=list_graphs[largest_ind]#p.load(open("../data/OASIS_full_batch/modified_graphs/graph_"+str(largest_ind)+".gpickle","rb"))
    color_label_ordered = gca.label_nodes_according_to_coord(g_l, mesh, coord_dim=1)
    r_perm=p.load(open("/mnt/data/work/python_sandBox/Graph_matching/data/r_perm.gpickle","rb"))
    color_label = color_label_ordered[r_perm]
    reg_mesh = gv.reg_mesh(mesh)
    vb_sc = gv.visbrain_plot(reg_mesh)

    default_value = -0.1#0.05
    nb_nodes = len(g_l.nodes)
    row_scope = range(largest_ind * nb_nodes, (largest_ind + 1) * nb_nodes)


    nb_unmatched = 0
    for i in range(nb_graphs):

        g = copy.deepcopy(list_graphs[i])
        #g=p.load(open("../data/OASIS_full_batch/modified_graphs/graph_"+str(i)+".gpickle","rb"))
        col_scope = range(i * nb_nodes, (i + 1) * nb_nodes)

        perm_X = np.array(matching_matrix[np.ix_(row_scope, col_scope)], dtype=int) #Iterate through each Perm Matrix X fixing the largest graph
        transfered_labels = np.ones(101)*default_value

        #nb_nodes = len(g.nodes)
        #col_scope = range(i * nb_nodes, (i + 1) * nb_nodes)

        for node_indx,ind in enumerate(row_scope):
            match_index = np.where(perm_X[node_indx,:]==1)[0]

            if len(match_index)>0:
                transfered_labels[match_index[0]] = color_label[node_indx]
        nb_unmatched += np.sum(transfered_labels==default_value)
        data_mask = gp.remove_dummy_nodes(g)
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(g, nodes_coords, node_data=transfered_labels[data_mask], nodes_mask=None, c_map='nipy_spectral')
        vb_sc.add_to_subplot(s_obj)
    print('nb_unmatched',nb_unmatched)
    print("Preview")



    print('get_clusters_from_assignment')
    gca.get_labelling_from_assignment(list_graphs, matching_matrix, largest_ind, mesh, label_attribute)
    print('create_clusters_lists')
    cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=label_attribute)
    # Calculate the centroid
    print('get_centroid_clusters')
    centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict)
    pickle_out = open(os.path.join(path_to_silhouette, label_attribute+'_silhouette.gpickle'), "rb")
    silhouette_dict = p.load(pickle_out)
    pickle_out.close()
    clust_silhouette = gca.get_silhouette_per_cluster(silhouette_dict)
    centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, mesh)
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=clust_silhouette,
                                                        nodes_size=60, nodes_mask=None, c_map='jet', symbol='disc',
                                                        vmin=-1, vmax=1)


    vb_sc.add_to_subplot(s_obj)
    vb_sc.preview()



        # for l in range(len(g_l)):
        #     index = np.where()
        #     transfered_labels[index]=color_label[l]




    # is_dummy = []
    # for i in range(nb_graphs):
    #     sing_graph = p.load(open("../data/OASIS_full_batch/modified_graphs/graph_"+str(i)+".gpickle","rb"))
    #     is_dummy.append(list(nx.get_node_attributes(sing_graph,"is_dummy").values()))
    #
    # is_dummy_vect = [val for sublist in is_dummy for val in sublist]
    #
    # # # Get the mesh
    # mesh = sio.load_mesh(template_mesh)
    # vb_sc = gv.visbrain_plot(mesh)
    #
    # for i in range(nb_graphs):
    #     match_label_per_graph={}
    #
    #     g = p.load(open("../data/OASIS_full_batch/modified_graphs/graph_"+str(i)+".gpickle","rb"))
    #     nb_nodes = len(g.nodes)
    #     scope = range(i * nb_nodes, (i + 1) * nb_nodes)
    #     for node_indx,ind in enumerate(scope):
    #         match_indexes = np.where(matching_matrix[ind,:]==1)[0]
    #         match_perc = (len(match_indexes) - len(set(match_indexes).intersection(np.where(np.array(is_dummy_vect)==True)[0])))/nb_graphs
    #         match_label_per_graph[node_indx] = {'label_color':match_perc}
    #
    #     nx.set_node_attributes(g, match_label_per_graph)
    #
    #     gp.remove_dummy_nodes(g)
    #
    #     nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
    #     node_data = gp.graph_nodes_attribute(g, "label_color")
    #     s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(g, nodes_coords, node_data=node_data, nodes_mask=None, c_map='nipy_spectral')#'rainbow')
    #     vb_sc.add_to_subplot(s_obj)
    #
    # vb_sc.preview()
    #


    # list_graphs = gp.load_graphs_in_list(path_to_graphs)
    # for g in list_graphs:
    #     gp.remove_dummy_nodes(g)
    #     print(len(g))

    # # Get the mesh
    # mesh = sio.load_mesh(template_mesh)
    # vb_sc = gv.visbrain_plot(mesh)
    # gp.remove_dummy_nodes(g)
    # label_nodes_according_to_coord(g, mesh, coord_dim=1)
    # nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
    # s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(g, nodes_coords, node_color_attribute="label_color", nodes_mask=None, c_map='nipy_spectral')#'rainbow')
    # vb_sc.add_to_subplot(s_obj)
    # vb_sc.preview()

    # for ind_g, g in enumerate(list_graphs):
    #     gp.remove_dummy_nodes(g)
    #     label_nodes_according_to_coord(g, mesh, coord_dim=1)
    #     nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
    #     node_data = gp.graph_nodes_attribute(g, "label_color")
    #     s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(g, nodes_coords, node_data=node_data, nodes_mask=None, c_map='nipy_spectral')#'rainbow')
    #     vb_sc.add_to_subplot(s_obj)

    # vb_sc.preview()
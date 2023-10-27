import sys
import os
#sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import tools.clusters_analysis as gca
import numpy as np
import networkx as nx
import scipy.io as sco
import pickle as p


if __name__ == "__main__":
    template_mesh = '../data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii'#lh.OASIS_testGrp_average_inflated.gii'
    reg_mesh = gv.reg_mesh(sio.load_mesh(template_mesh))
    c_map = 'nipy_spectral'
    path_to_X = "../data/Oasis_original_new_with_dummy"
    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    reg_or_unreg = ''#'_unreg'#''
    method = 'MatchEig'#'mALS'#'neuroimage'#'mSync'#'kerGM'#'CAO'#'media'#'mALS'#'kmeans_70_real_data_dummy'#'media'#'CAO'#'mALS'#
    default_label = -0.1
    vmin = -0.1
    vmax = 1.1
#     vmin = 0
#     vmax = 300

    label_to_plot = 28#-2#default_label#222


    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    #list_graphs = list_graphs_i[3:5]
    print('----------------------------')
    print(method)
    # compute the labeling from the assignment matrix when needed
    if 'media' in method:
        label_attribute = 'label_media'
    elif 'neuroimage' in method:
        label_attribute = 'label_neuroimage'
    else:
        largest_ind=22
        label_attribute = 'labelling_from_assgn'
        # load the assignment matrix
        file_X = os.path.join(path_to_X, "X_" + method + reg_or_unreg + ".mat")
        if 'kerGM' in method:
            X = sco.loadmat(file_X)["full_assignment_mat"]
        else:
            X = sco.loadmat(file_X)['X']
        trans_l = gca.get_labelling_from_assignment(list_graphs, X, largest_ind, reg_mesh, label_attribute, default_label_value=default_label)

    print('create_clusters_lists')
    cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=label_attribute)
    print(len(cluster_dict))
    print(cluster_dict.keys())
    # Calculate the centroid
    print('get_centroid_clusters')
    centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict, coords_attribute="sphere_3dcoords")

    vb_sc = gv.visbrain_plot(reg_mesh)
    tot_nb_nodes = 0
    len_graphs = list()
    for i, g in enumerate(list_graphs):
        gp.remove_dummy_nodes(g)
        len_graphs.append(len(g))
        tot_nb_nodes += len(g)

    for g in list_graphs:
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
        #labels = nx.get_node_attributes(g, 'label_media').values()
        labels = nx.get_node_attributes(g, label_attribute).values()
        color_label = np.array([l for l in labels])
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=color_label, nodes_mask=None,
                                                        c_map=c_map,  vmin=vmin, vmax=vmax)
        vb_sc.add_to_subplot(s_obj)

    centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, reg_mesh, attribute_vertex_index='ico100_7_vertex_index')
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=np.array(list(cluster_dict.keys())),
                                                        nodes_size=90, nodes_mask=None, c_map=c_map, symbol='disc',
                                                        vmin=vmin, vmax=vmax)

    vb_sc.add_to_subplot(s_obj)
    vb_sc.preview()


    # # plot a specific label only
    # vb_sc2 = gv.visbrain_plot(reg_mesh)
    #
    # for ind,g in enumerate(list_graphs):
    #
    #     nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
    #     labels = nx.get_node_attributes(g, label_attribute).values()
    #     color_label = np.array([l for l in labels])
    #     color_label_to_plot = np.ones(color_label.shape)
    #     color_label_to_plot[color_label == label_to_plot]=0
    #     #print(color_label)
    #     if np.sum(color_label == label_to_plot)==0:
    #         print(ind)
    #     else:
    #         s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=color_label_to_plot, nodes_mask=None, c_map='nipy_spectral',  vmin=0, vmax=1)
    #         vb_sc2.add_to_subplot(s_obj)
    # vb_sc2.preview()


    # default_value = -0.1#0.05
    # nb_nodes = len(g_l.nodes)c_map='nipy_spectral'
    #
    #
    #
    # for j in range(len(list_graphs)):
    #
    #     grph = list_graphs[j]
    #     nodes_to_remove = gp.remove_dummy_nodes(grph)
    #     nodes_to_remove = np.where(np.array(nodes_to_remove)==False)
    #
    #     grph.remove_nodes_from(list(nodes_to_remove[0]))
    #     nb_nodes = len(grph.nodes)
    #     row_scope = range(j * nb_nodes, (j + 1) * nb_nodes)
    #
    #     print(len(grph.nodes))
    #
    #     if len(grph.nodes)==101:
    #         break
    #
    #
    # for matching_matrix in X:
    #
    #     print(matching_matrix.shape)
    #     last_index = 0
    #
    #     nb_unmatched = 0
    #     for i in range(nb_graphs):
    #
    #         #g = list_graphs[i]
    #         g=p.load(open("../data/OASIS_full_batch/modified_graphs/graph_"+str(i)+".gpickle","rb"))
    #
    #         nodes_to_remove = gp.remove_dummy_nodes(g)
    #         nodes_to_remove = np.where(np.array(nodes_to_remove)==False)
    #         g.remove_nodes_from(list(nodes_to_remove[0]))
    #         nb_nodes = len(g.nodes)
    #
    #         print(len(g.nodes))
    #
    #         if i == 0:
    #             col_scope = range(i * nb_nodes, (i + 1) * nb_nodes)
    #             prev_nb_nodes = nb_nodes
    #             perm_X = np.array(matching_matrix[np.ix_(row_scope, col_scope)], dtype=int) #Iterate through each Perm Matrix X fixing the largest graph
    #             transfered_labels = np.ones(nb_nodes)*default_value
    #             last_index+=nb_nodes
    #         else:
    #             col_scope = range(last_index, last_index+nb_nodes)
    #             last_index += nb_nodes
    #             perm_X = np.array(matching_matrix[np.ix_(row_scope, col_scope)], dtype=int) #Iterate through each Perm Matrix X fixing the largest graph
    #             transfered_labels = np.ones(nb_nodes)*default_value
    #
    #
    #         print(col_scope)
    #
    #         #nb_nodes = len(g.nodes)
    #         #col_scope = range(i * nb_nodes, (i + 1) * nb_nodes)
    #
    #         for node_indx,ind in enumerate(row_scope):
    #             match_index = np.where(perm_X[node_indx,:]==1)[0]
    #
    #             if len(match_index)>0:
    #                 transfered_labels[match_index[0]] = color_label[node_indx]
    #         nb_unmatched += np.sum(transfered_labels==default_value)
    #         #data_mask = gp.remove_dummy_nodes(g)
    #         nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
    #         print(nodes_coords.shape)
    #         s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=transfered_labels, nodes_mask=None, c_map='nipy_spectral', symbol='disc', vmin=0, vmax=1)
    #
    #
    #         vb_sc.add_to_subplot(s_obj)
    #     print('nb_unmatched',nb_unmatched)
    #     print("Preview")
    #     vb_sc.preview()



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
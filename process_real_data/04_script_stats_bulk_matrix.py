import sys

#sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import numpy as np
import networkx as nx
import scipy.io as sco
import pickle as p
import matplotlib.pyplot as plt
import os
from visbrain.objects import SourceObj, ColorbarObj

if __name__ == "__main__":
    c_map = 'hot'
    vmin = 80
    vmax = 100
    template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    reg_mesh = gv.reg_mesh(sio.load_mesh(template_mesh))
    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    path_to_X = '../data/Oasis_original_new_with_dummy'

    reg_or_unreg = ''#'_unreg'#''
    methods = ['media', 'media_no_excl']#,'neuroimage', 'mALS', 'kmeans_70_real_data', 'mSync']#,'kerGM', 'CAO']#['media', 'neuroimage']#,
    #methods = ['kmeans_70_real_data_dummy','kmeans_90_real_data_dummy','kmeans_110_real_data_dummy']
    # load the graphs
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    nb_graphs = len(list_graphs)
    print('nb graphs ', nb_graphs)
    # compute the mask of dummy nodes
    is_dummy_vect = []
    for g in list_graphs:
        is_dummy_vect.extend(list(nx.get_node_attributes(g, "is_dummy").values()))
    not_dummy_vect = np.logical_not(is_dummy_vect)
    print('nb nodes per graph (incl. dummy nodes ', len(g))
    print('total nb of nodes ', len(g)*nb_graphs)
    print(len(is_dummy_vect))#_vect))
    print(len(not_dummy_vect))
    print('total nb of dummy nodes ', np.sum(is_dummy_vect))
    print('total nb of non-dummy nodes', np.sum(not_dummy_vect))



    nb_bins = 50
    dens = False
    fig1, ax = plt.subplots(2, len(methods), sharey=True)

    for ind, method in enumerate(methods):
        print('----------------------------')
        print(method)
        # load the assignment matrix
        if ('kmeans' in method) or ('neuroimage' in method) or ('media' in method):
            file_X = os.path.join(path_to_X, "X_" + method + reg_or_unreg + "_dummy.mat")
        else:
            file_X = os.path.join(path_to_X, "X_" + method + reg_or_unreg + ".mat")
        if ('kerGM' in method) or ('kmeans' in method) or ('neuroimage' in method) or ('media' in method):
            X = sco.loadmat(file_X)["full_assignment_mat"]
        else:
            X = sco.loadmat(file_X)['X']
        #print(X.shape)

        # compute for each row of the assignment matrix the percent of matched nodes across the graphs
        match_no_dummy = np.sum(X[:, not_dummy_vect], 1)
        match_dummy = np.sum(X[:, is_dummy_vect], 1)
        print(max(match_no_dummy))
        print(max(match_dummy))
        # plot the ditribution across the rows of the matrix
        #ax[ind].hist(match_no_dummy, density=dens, bins=nb_bins)  # density=False would make counts
        absc = range(len(match_no_dummy))
        ax[0, ind].bar(absc, np.sort(match_no_dummy))  # density=False would make counts
        ax[0, ind].set_ylabel('percent of total nb of nodes per label')
        ax[0, ind].set_xlabel('Data')
        ax[0, ind].set_title('real match for '+method)
        ax[1, ind].bar(absc, np.sort(match_dummy))  # density=False would make counts
        ax[1, ind].set_ylabel('percent of total nb of nodes per label')
        ax[1, ind].set_xlabel('Data')
        ax[1, ind].set_title('dummy match for '+method)

    plt.show()

    # # visu on the mesh
    # # # Get the mesh
    # mesh = gv.reg_mesh(sio.load_mesh(template_mesh))
    # vb_sc = gv.visbrain_plot(mesh)
    # vb_sc = gv.visbrain_plot(mesh, caption='mSync')
    # vb_sc2 = gv.visbrain_plot(mesh, caption='mALS')
    # vb_sc3 = gv.visbrain_plot(mesh, caption='mKerGM')
    # #vb_sc4 = gv.visbrain_plot(mesh, caption='mHippi')
    #
    # for i in range(nb_graphs):
    #     g=list_graphs[i]
    #     #match_label_per_graph = {}
    #     nb_nodes = len(g.nodes)
    #     scope = range(i * nb_nodes, (i + 1) * nb_nodes)
    #     data_match_dummy_mSync = match_dummy_mSync[scope]
    #     data_match_no_dummy_mSync = match_no_dummy_mSync[scope]
    #     data_match_dummy_mALS = match_dummy_mALS[scope]
    #     data_match_no_dummy_mALS = match_no_dummy_mALS[scope]
    #     data_match_dummy_kerGM = match_dummy_kerGM[scope]
    #     data_match_no_dummy_kerGM = match_no_dummy_kerGM[scope]
    #
    #     data_mask = gp.remove_dummy_nodes(g)
    #
    #     print(np.min(data_match_no_dummy_mSync[data_mask]))
    #     print(np.max(data_match_no_dummy_mSync[data_mask]))
    #
    #     nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
    #     # s_obj, cb_obj = show_graph_nodes(g, mesh, data=data_match_no_dummy_mSync[data_mask], clim=clim)
    #     s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=data_match_no_dummy_mSync[data_mask],
    #                                                     nodes_mask=None, c_map=c_map, symbol='disc',
    #                                                     vmin=vmin, vmax=vmax)
    #     vb_sc.add_to_subplot(s_obj)
    #
    #     #s_obj, cb_obj = show_graph_nodes(g, mesh, data=data_match_no_dummy_mALS[data_mask], clim=clim)
    #     s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=data_match_no_dummy_mALS[data_mask],
    #                                                     nodes_mask=None, c_map=c_map, symbol='disc',
    #                                                     vmin=vmin, vmax=vmax)
    #     vb_sc2.add_to_subplot(s_obj)
    #
    #     #s_obj, cb_obj = show_graph_nodes(g, mesh, data=data_match_no_dummy_kerGM[data_mask], clim=clim)
    #     s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=data_match_no_dummy_kerGM[data_mask],
    #                                                     nodes_mask=None, c_map=c_map, symbol='disc',
    #                                                     vmin=vmin, vmax=vmax)
    #     vb_sc3.add_to_subplot(s_obj)
    #
    # # visb_sc_shape = gv.get_visb_sc_shape(vb_sc)
    # # vb_sc.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
    # #                           col=3, width_max=200)
    #
    #     # vb_sc2 = gv.visbrain_plot(mesh)
    #     # s_obj, cb_obj = show_graph_nodes(g, mesh, data=data_match_dummy_mSync[data_mask], clim=clim)
    #     # visb_sc_shape = gv.get_visb_sc_shape(vb_sc2)
    #     # vb_sc2.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1] - 1)
    #     # vb_sc2.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
    #     #                      col=visb_sc_shape[1] + 1, width_max=200)
    #     # vb_sc2.preview()
    # # curr_node=0
    # # for i in range(nb_graphs):
    # #     g = list_graphs[i]
    # #     gp.remove_dummy_nodes(g)
    # #     #match_label_per_graph = {}
    # #     nb_nodes = len(g.nodes)
    # #     print(nb_nodes)
    # #     scope = range(curr_node, curr_node+nb_nodes)
    # #     curr_node = curr_node+nb_nodes
    # #
    # #     data_match_no_dummy_Hippi = match_no_dummy_Hippi[scope]
    # #     s_obj, cb_obj = show_graph_nodes(g, mesh, data=data_match_no_dummy_Hippi, clim=clim)
    # #     vb_sc4.add_to_subplot(s_obj)
    # #
    # # visb_sc_shape = gv.get_visb_sc_shape(vb_sc3)
    # # vb_sc3.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
    # #                           col=visb_sc_shape[1] + 1, width_max=200)
    # vb_sc.preview()
    # vb_sc2.preview()
    # vb_sc3.preview()
    # #vb_sc4.preview()



    # list_graphs = gp.load_graphs_in_list(path_to_graphs)
    # for g in list_graphs:
    #     gp.remove_dummy_nodes(g)
    #     print(len(g))

    # # Get the mesh
    # mesh = sio.load_mesh(template_mesh)
    # vb_sc = gv.visbrain_plot(mesh)
    # # gp.remove_dummy_nodes(g)
    # # label_nodes_according_to_coord(g, mesh, coord_dim=1)
    # # nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
    # # s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(g, nodes_coords, node_color_attribute="label_color", nodes_mask=None, c_map='nipy_spectral')#'rainbow')
    # # vb_sc.add_to_subplot(s_obj)
    # # vb_sc.preview()

    # for ind_g, g in enumerate(list_graphs):
    #     gp.remove_dummy_nodes(g)
    #     label_nodes_according_to_coord(g, mesh, coord_dim=1)
    #     nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
    #     s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(g, nodes_coords, node_color_attribute="label_color", nodes_mask=None, c_map='nipy_spectral')#'rainbow')
    #     vb_sc.add_to_subplot(s_obj)

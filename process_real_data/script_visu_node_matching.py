import sys
sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import numpy as np
import networkx as nx
import scipy.io as sco
import pickle as p
import matplotlib.pyplot as plt
import os

def label_nodes_according_to_coord(graph_no_dummy, template_mesh, coord_dim=1):
    nodes_coords = gp.graph_nodes_to_coords(graph_no_dummy, 'ico100_7_vertex_index', template_mesh)
    one_nodes_coords = nodes_coords[:, coord_dim]
    one_nodes_coords_scaled = (one_nodes_coords - np.min(one_nodes_coords))/(np.max(one_nodes_coords)-np.min(one_nodes_coords))
    # initialise the dict for atttributes
    nodes_attributes = {}
    # Fill the dictionnary with the nd_array attribute
    for ind, node in enumerate(graph_no_dummy.nodes):
        nodes_attributes[node] = {"label_color": one_nodes_coords_scaled[ind]}

    nx.set_node_attributes(graph_no_dummy, nodes_attributes)

def show_graph_nodes(graph, mesh, data, clim=(0, 1), transl=None):

    # manage nodes
    s_coords = gp.graph_nodes_to_coords(graph, 'ico100_7_vertex_index', mesh)
    print("s_coords",s_coords.shape)

    transl_bary = np.mean(s_coords)
    s_coords = 1.01*(s_coords-transl_bary)+transl_bary

    if transl is not None:
        s_coords += transl

    s_obj = SourceObj('nodes', s_coords, color='red',#data=data[data_mask],
                        edge_color='black', symbol='disc', edge_width=2.,
                        radius_min=30., radius_max=30., alpha=.9)
    """Color the sources according to data
    """
    s_obj.color_sources(data=data, cmap='hot', clim=clim)
    # Get the colorbar of the source object
    CBAR_STATE = dict(cbtxtsz=30, txtsz=30., width=.1, cbtxtsh=3.,
                          rect=(-.3, -2., 1., 4.), txtcolor='k')
    cb_obj = ColorbarObj(s_obj, cblabel='node consistency', border=False,
                  **CBAR_STATE)

    return s_obj, cb_obj


if __name__ == "__main__":
    #template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/ico100_7.gii'
    template_mesh = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/OASIS_full_batch/modified_graphs'

    Hippi_path = '/home/rohit/PhD_Work/GM_my_version/RESULT_FRIOUL_HIPPI/Hippi_res_real_mat.npy'



    path_to_match_mat = "/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/OASIS_full_batch"

    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    x_mSync = sco.loadmat(os.path.join(path_to_match_mat,"X_mSync.mat"))["X"]
    x_mALS = sco.loadmat(os.path.join(path_to_match_mat,"X_mALS.mat"))["X"]
    x_cao = sco.loadmat(os.path.join(path_to_match_mat,"X_cao_cst_o.mat"))["X"]
    Hippi = np.load(Hippi_path)
    x_Kergm = sco.loadmat(os.path.join(path_to_match_mat,"X_pairwise_kergm.mat"))["full_assignment_mat"]


    clim=(0, 1)
    matching_matrix = Hippi
    nb_graphs = 134

    is_dummy = []
    for i in range(nb_graphs):
        sing_graph = p.load(open("../data/OASIS_full_batch/modified_graphs/graph_"+str(i)+".gpickle","rb"))
        is_dummy.append(list(nx.get_node_attributes(sing_graph,"is_dummy").values()))
    
    is_dummy_vect = [val for sublist in is_dummy for val in sublist]

    print("len is_dummy_vect",len(is_dummy_vect))
    print("shape matching mat",matching_matrix.shape)

    # # Get the mesh
    mesh = sio.load_mesh(template_mesh)
    vb_sc = gv.visbrain_plot(mesh)

    for i in range(nb_graphs):
        match_label_per_graph={}
    
        g = p.load(open("../data/OASIS_full_batch/modified_graphs/graph_"+str(i)+".gpickle","rb"))
        nb_nodes = len(g.nodes)
        gp.remove_dummy_nodes(g)
        scope = range(i * nb_nodes, (i + 1) * nb_nodes)
        for node_indx,ind in enumerate(scope):
            match_indexes = np.where(matching_matrix[ind,:]==1)[0]
            match_perc = (len(match_indexes) - len(set(match_indexes).intersection(np.where(np.array(is_dummy_vect)==True)[0])))/nb_graphs
            match_label_per_graph[node_indx] = {'label_color':match_perc}
        
        nx.set_node_attributes(g, match_label_per_graph)
        data_mask = gp.remove_dummy_nodes(g)

        his_data = list(nx.get_node_attributes(g,'label_color').values())

        plt.hist(his_data, density=False, bins=50) # density=False would make counts
        plt.ylabel('Frequency')
        plt.xlabel('Data')
        plt.title('For 1 graph: number of nodes matched across graphs by Hippi')
        plt.show()



        #nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
        node_data = gp.graph_nodes_attribute(g, "label_color")
        s_obj, cb_obj = show_graph_nodes(g, mesh, data=node_data[data_mask], clim=clim)
        visb_sc_shape = gv.get_visb_sc_shape(vb_sc)
        vb_sc.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
    vb_sc.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
                           col=visb_sc_shape[1] + 1, width_max=200)
    vb_sc.preview()
        


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

    # vb_sc.preview()
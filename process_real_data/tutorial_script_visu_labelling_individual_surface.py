import sys
sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])

import slam.io as sio
import networkx as nx
import tools.graph_visu as gv
import tools.graph_processing as gp
import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":

    #file_template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    # file_mesh = '../data/example_individual_OASIS_0061/lh.white.gii'
    # file_basins = '../data/example_individual_OASIS_0061/alpha0.03_an0_dn20_r1.5_L_area50FilteredTexture.gii'
    # file_graph = '../data/example_individual_OASIS_0061/OAS1_0061_lh_pitgraph.gpickle'

    file_mesh = '../data/Oasis_all_subjects_white/FS_OASIS/OAS1_0439/surf/lh.white.gii'
    file_basins = '../data/OASIS_all_subjects/OAS1_0439/dpfMap/alpha_0.03/an0_dn20_r1.5/alpha0.03_an0_dn20_r1.5_L_area50FilteredTexture.gii'

    path_to_labelled_graphs = '../data/Oasis_original_new_with_dummy/labelled_graphs/'



    list_graphs = gp.load_graphs_in_list(path_to_labelled_graphs)

    #graph = nx.read_gpickle(file_graph)
    #gp.preprocess_graph(graph)

    graph = list_graphs[56]  # corresponds to nth subject

    # TUTO 1 :: plot the graph on corresponding individual cortical mesh
    # load the mesh
    mesh = sio.load_mesh(file_mesh)
    # eventually smooth it a bit
    # import trimesh.smoothing as tms
    # mesh = tms.filter_laplacian(mesh, iterations=80)
    # load the basins texture
    tex_basins = sio.load_texture(file_basins)

    # modified_tex = tex_basins.darray[0]
    # modified_tex[tex_basins.darray[0]>354]=0
    # modified_tex[tex_basins.darray[0]<354]=0

    # # plot the mesh with basin texture

    # vb_sc = gv.visbrain_plot(mesh, tex=tex_basins.darray[2], cmap='tab20c',
    #                          caption='Visu on individual mesh with basins',
    #                          cblabel='basins colors')

    
    gp.remove_dummy_nodes(graph)
    
    matching_labels_tex = np.zeros_like(tex_basins.darray[0].copy())
    for n in graph.nodes:
        lab_matching_value = graph.nodes.data()[n]['labelling_mSync']  
        basin_tex_label = graph.nodes.data()[n]['basin_label']
        matching_labels_tex[tex_basins.darray[0] == basin_tex_label] = lab_matching_value



    vb_sc2 = gv.visbrain_plot(mesh, tex=matching_labels_tex, cmap='jet',
                             caption='Visu on individual mesh with basins',
                             cblabel='basins colors')

    nodes_coords = gp.graph_nodes_to_coords(graph, 'vertex_index', mesh)

    s_obj2, c_obj2, node_cb_obj2 = gv.show_graph(graph, nodes_coords, node_color_attribute='labelling_mSync',
                                              edge_color_attribute=None,
                                              nodes_mask=None,c_map='jet')
    #vb_sc2.add_to_subplot(c_obj2)
    vb_sc2.add_to_subplot(s_obj2)
    visb_sc_shape = gv.get_visb_sc_shape(vb_sc2)
    vb_sc2.add_to_subplot(node_cb_obj2, row=visb_sc_shape[0] - 1, col=visb_sc_shape[0] + 0, width_max=200)

    # show the plot on the screen
    vb_sc2.preview()

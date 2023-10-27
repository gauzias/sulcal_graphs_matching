import sys
sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])

import slam.io as sio
import networkx as nx
import tools.graph_visu as gv
import tools.graph_processing as gp



if __name__ == "__main__":

    file_template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    file_mesh = '../data/example_individual_OASIS_0061/rh.white.gii'
    file_basins = '../data/example_individual_OASIS_0061/alpha0.03_an0_dn20_r1.5_R_area50FilteredTexture.gii'
    file_graph = '../data/example_individual_OASIS_0061/OAS1_0061_rh_pitgraph.gpickle'

    graph = nx.read_gpickle(file_graph)
    gp.preprocess_graph(graph)

    # TUTO 1 :: plot the graph on corresponding individual cortical mesh
    # load the mesh
    mesh = sio.load_mesh(file_mesh)
    # eventually smooth it a bit
    # import trimesh.smoothing as tms
    # mesh = tms.filter_laplacian(mesh, iterations=80)
    # load the basins texture
    tex_basins = sio.load_texture(file_basins)
    # plot the mesh with basin texture
    vb_sc = gv.visbrain_plot(mesh, tex=tex_basins.darray[2], cmap='tab20c',
                             caption='Visu on individual mesh with basins',
                             cblabel='basins colors')
    # process the graph
    # 1 remove potential dummy nodes (which are not connected by any edge and have no coordinate)
    gp.remove_dummy_nodes(graph)
    # 2 compute nodes coordinates in 3D by retrieving the mesh vertex corresponding to each graph node, based on the
    # corresponding node attribute
    nodes_coords = gp.graph_nodes_to_coords(graph, 'vertex_index', mesh)
    # 3 eventually compute a mask for masking some part of the graph
    mask_slice_coord=15
    nodes_mask = nodes_coords[:, 1] > mask_slice_coord
    # 4 create the objects for visualization of the graph and add these to the figure
    s_obj, c_obj, node_cb_obj = gv.show_graph(graph, nodes_coords, node_color_attribute=None, edge_color_attribute=None,
                                 nodes_mask=nodes_mask)
    vb_sc.add_to_subplot(c_obj)
    vb_sc.add_to_subplot(s_obj)

    # show the plot on the screen
    vb_sc.preview()


    # TUTO 2 :: visualize nodes and edges attributes
    # use the same individual mesh
    vb_sc2 = gv.visbrain_plot(mesh, caption='Visu on individual mesh')
    s_obj2, c_obj2, node_cb_obj2 = gv.show_graph(graph, nodes_coords, node_color_attribute='depth',
                                              edge_color_attribute='geodesic_distance',
                                              nodes_mask=None)
    vb_sc2.add_to_subplot(c_obj2)
    vb_sc2.add_to_subplot(s_obj2)
    visb_sc_shape = gv.get_visb_sc_shape(vb_sc2)
    vb_sc2.add_to_subplot(node_cb_obj2, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1] + 1, width_max=200)

    # show the plot on the screen
    vb_sc2.preview()


    # TUTO 3 :: plot the graph on the template mesh
    # load and reorient the template mesh
    template_mesh = gv.reg_mesh(sio.load_mesh(file_template_mesh))

    # plot the mesh with basin texture
    vb_sc3 = gv.visbrain_plot(template_mesh, caption='Visu on template mesh')
    # process the graph
    # 1 remove potential dummy nodes (which are not connected by any edge and have no coordinate)
    gp.remove_dummy_nodes(graph)
    # 2 compute nodes coordinates in 3D by retrieving the mesh vertex corresponding to each graph node, based on the
    # corresponding node attribute
    nodes_coords = gp.graph_nodes_to_coords(graph, 'ico100_7_vertex_index', template_mesh)
    # 3 eventually compute a mask for masking some part of the graph
    mask_slice_coord = -15
    nodes_mask = nodes_coords[:, 2] > mask_slice_coord
    # 4 create the objects for visualization of the graph and add these to the figure
    s_obj3, c_obj3, node_cb_obj3 = gv.show_graph(graph, nodes_coords, node_color_attribute='basin_thickness',
                                                edge_color_attribute='geodesic_distance',
                                                nodes_mask=nodes_mask)
    vb_sc3.add_to_subplot(c_obj3)
    vb_sc3.add_to_subplot(s_obj3)
    visb_sc_shape = gv.get_visb_sc_shape(vb_sc3)
    vb_sc3.add_to_subplot(node_cb_obj3, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1] + 1, width_max=200)

    # show the plot on the screen
    vb_sc3.preview()
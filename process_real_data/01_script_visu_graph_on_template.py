import sys
sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import numpy as np


if __name__ == "__main__":
    template_mesh = '../data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii'#lh.OASIS_testGrp_average_inflated.gii'
    mesh = gv.reg_mesh(sio.load_mesh(template_mesh))
    # template_mesh = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    # path_to_graphs = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/OASIS_full_batch/modified_graphs/'
    path_to_graphs = '../data/Oasis_original_new/'
    file_sphere_mesh = '../data/template_mesh/ico100_7.gii'
    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    # Get the mesh
    sphere_mesh = sio.load_mesh(file_sphere_mesh)
    mask_slice_coord = -15
    vb_sc = None
    inds_to_show = [0,1,11]
    graphs_to_show=[list_graphs[i] for i in inds_to_show]
    for g in graphs_to_show:
        # gp.sphere_nearest_neighbor_interpolation(g, sphere_mesh)
        #
        #nodes_to_remove = gp.remove_dummy_nodes(g)
        # nodes_to_remove = np.where(np.array(nodes_to_remove)==False)
        # g.remove_nodes_from(list(nodes_to_remove[0]))

        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
        #nodes_mask = nodes_coords[:,2]>mask_slice_coord
        s_obj, c_obj, node_cb_obj = gv.show_graph(g, nodes_coords,node_color_attribute=None, nodes_size=30, c_map='nipy_spectral')
        vb_sc = gv.visbrain_plot(mesh, visb_sc=vb_sc)
        visb_sc_shape = gv.get_visb_sc_shape(vb_sc)

        vb_sc.add_to_subplot(c_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
        vb_sc.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1] - 1)

    # s_obj, c_obj = show_graph(list_graphs[0], mesh, edge_attribute='geodesic_distance')
    # vb_sc = visbrain_plot(mesh)
    # vb_sc.add_to_subplot(s_obj)
    # vb_sc.add_to_subplot(c_obj)
    vb_sc.preview()


    sphere_mesh = sio.load_mesh(file_sphere_mesh)
    vb_sc2 = gv.visbrain_plot(sphere_mesh)
    for g in list_graphs:
        nodes_coords = gp.graph_nodes_to_coords(g,  'ico100_7_vertex_index', sphere_mesh)
        #nodes_mask = nodes_coords[:,2]>mask_slice_coord
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords)

        vb_sc2.add_to_subplot(s_obj)

    vb_sc2.preview()


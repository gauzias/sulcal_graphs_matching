import sys
sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp


if __name__ == "__main__":
    template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    #                '/mnt/data/work/python_sandBox/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/OASIS_full_batch/modified_graphs'

    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    for g in list_graphs:
        print(len(g))

    # Get the mesh
    mesh = sio.load_mesh(template_mesh)
    vb_sc = gv.visbrain_plot(mesh)
    for ind_g, g in enumerate(list_graphs):
        gp.remove_dummy_nodes(g)
        s_obj = gv.graph_nodes_coords_to_sources(g)
        vb_sc.add_to_subplot(s_obj)
    vb_sc.preview()

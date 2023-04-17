import matplotlib.pyplot as plt
import numpy as np


def myGraphPlot(G, figsize=(10,5), coords="ring2D", title=None, vertex_names=None, print_graph_info=False, display_edgeweights=False, display_vertexnames=False, ax=None):
    r"""
        Customized Plot function for GraphPlot of PyGsp
    """
    plt_show = False
    if (ax is None):
        fig, ax = plt.subplots(1,1,figsize=(10,5))
        plt_show=True
    
    try:
        getattr(G, "coords")
        coords = G.coords
    except:
        G.set_coordinates(coords)
        
    G.plot(ax=ax)
    ax.set_axis_off()
    
    W = G.W.toarray()

    if vertex_names:
        vertices = vertex_names
    else:
        vertices = np.arange(G.W.shape[0])+1
        
    if title:
        pass
    else:
        title = coords if isinstance(coords, str) else "Unknown"
    title += "\n G.N=" + str(G.N) + " nodes, G.Ne=" + str(G.Ne) + " edges\n"
    ax.set_title(title, fontsize=14, fontweight=600)
        
    if display_vertexnames:
        if (isinstance(coords, str) == False):
            coords = np.array(coords)
            for i, coord in enumerate(coords):
                ax.text(coord[0]-0.1, coord[1]+0.2, vertices[i], fontsize=20, fontweight=600)
        
        
        
    if display_edgeweights:
        inds = np.where(W!=0)
        inds = list(zip(inds[0], inds[1]))
        for cord in inds:
            try:
                inds.remove((cord[1], cord[0]))
            except:
                pass
        for cord in inds:
            cord_x = (coords[cord[0]][0] + coords[cord[1]][0])/2
            cord_y = (coords[cord[0]][1] + coords[cord[1]][1])/2
        
            ax.text(cord_x-0.1, cord_y+0.2, W[cord], fontsize=16, fontweight=500)

    
    if plt_show:
        plt.show()


    if print_graph_info:
        print("V - set of vertices :")
        print(vertices, "\n")
        
        inds = np.where(W!=0)
        inds = list(zip(inds[0], inds[1]))
        print("E - set of edges : ")
        print(inds, "\n")
        
        print("W - weighted adjacency matrix :")
        print(W, "\n")
    
    

def myGraphPlotSignal(G, s, highlight=None, figsize=(10,5), coords="ring2D", title=None, vertex_names=None, print_graph_info=False, display_edgeweights=False, display_vertexnames=False, ax=None):
    r"""
        Customized Plot function for GraphPlotSignal of PyGsp
    """
    plt_show = False
    if (ax is None):
        fig, ax = plt.subplots(1,1,figsize=(10,5))
        plt_show=True
    try:
        getattr(G, "coords")
        coords = G.coords
    except:
        G.set_coordinates(coords)
        
    G.plot_signal(s, ax=ax)
    ax.set_axis_off()
    
    W = G.W.toarray()

    if vertex_names:
        vertices = vertex_names
    else:
        vertices = np.arange(G.W.shape[0])+1
        
    if title:
        pass
    else:
        title = coords if isinstance(coords, str) else "Unknown"
    title += "\n G.N=" + str(G.N) + " nodes, G.Ne=" + str(G.Ne) + " edges\n"
    ax.set_title(title, fontsize=14, fontweight=600)
        
    if display_vertexnames:
        if (isinstance(coords, str) == False):
            coords = np.array(coords)
            for i, coord in enumerate(coords):
                ax.text(coord[0]-0.1, coord[1]+0.2, vertices[i], fontsize=20, fontweight=600)
        
    if display_edgeweights:
        inds = np.where(W!=0)
        inds = list(zip(inds[0], inds[1]))
        for cord in inds:
            try:
                inds.remove((cord[1], cord[0]))
            except:
                pass
        for cord in inds:
            cord_x = (coords[cord[0]][0] + coords[cord[1]][0])/2
            cord_y = (coords[cord[0]][1] + coords[cord[1]][1])/2
        
            ax.text(cord_x-0.1, cord_y+0.2, W[cord], fontsize=16, fontweight=500)

    if plt_show:
        plt.show()


    if print_graph_info:
        print("V - set of vertices :")
        print(vertices, "\n")
        
        inds = np.where(W!=0)
        inds = list(zip(inds[0], inds[1]))
        print("E - set of edges : ")
        print(inds, "\n")
        
        print("W - weighted adjacency matrix :")
        print(W, "\n")
    
    


.. Pycsou-GSP documentation master file, created by
   sphinx-quickstart on Wed Apr 12 15:46:01 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Pycsou-GSP's documentation!
======================================
   
Check out the :doc:`installation` section for further information to
:ref:`install <installation>` the project.

.. plot::
    
    from pygsp import graphs
    from pycgsp.linop import diff
    from pycgsp.core import plot
    import numpy as np
    import matplotlib.pyplot as plt
    G = graphs.Minnesota()
    signal = np.random.randn(G.N,)
    #Lap = diff.GraphLaplacian(G)
    #lap_sig = Lap(signal)
    #lap_sig_pygsp = G.L.dot(signal)
    #fig, ax = plt.subplots(1, 3, figsize=(20,8))
    #plot.myGraphPlotSignal(G, s = signal, title="Input Signal", ax = ax[0])
    #plot.myGraphPlotSignal(G, s = lap_sig, title="Laplacian of Signal by Pycsou-GSP", ax = ax[1])
    #plot.myGraphPlotSignal(G, s = lap_sig_pygsp, title="Laplacian of Signal by PyGSP", ax = ax[2])
    #plt.show()
   

.. toctree::
    :maxdepth: 2
    :caption: Getting Started
    :hidden:

    installation
    theory/index
    
.. toctree::
   :maxdepth: 2
   :caption: Reference Documentation
   :hidden:
   
   api/index


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

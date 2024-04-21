import bokeh
import numpy as np

from typing import *
from numpy.typing import ArrayLike
from scipy.sparse import csc_matrix
from bokeh.plotting import figure
from bokeh.models import GraphRenderer, Circle, ColumnDataSource, MultiLine, Slider
from bokeh.io import output_notebook, show, curdoc
from bokeh.core.validation import check_integrity
from bokeh.layouts import column, row, gridplot

# %% 
class BokehNerveGraph():
  """ Class for drawing Nerve Graphs w/ Bokeh """
  def __init__(self, cover: csc_matrix, layout: Optional[ArrayLike] = None):
    #output_notebook(verbose=False, hide_banner=True)
    v_sizes = np.ravel(cover.astype(bool).sum(axis=0))
    
    ## Deflate empty opens
    self.cover = cover[:,v_sizes > 0] if np.any(v_sizes == 0) else cover 

    ## Construct the relations of nerve
    cover_bool = cover.astype(bool) # cast to boolean to treat non-zeros as indicators
    self.G = G = (cover_bool.astype(int).T @ cover_bool.astype(int))
    G.setdiag(0)

    ## Use force-directed to get initial layout     
    if layout is None: 
      import networkx as nx 
      layout = np.array(list(nx.spring_layout(nx.Graph(G)).values()))

    ## Parameterize the figure 
    fig_args = dict(title="Nerve visualization", tools="pan,wheel_zoom,lasso_select,reset", active_scroll=None, active_drag="auto", plot_width=500, plot_height=500)
    fig_args['tooltips'] = [("index", "$index"),("# elements", "@n")]
    fig_args['x_range'] = np.array([np.min(layout[:,0]), np.max(layout[:,0])])*[0.90, 1.10]
    fig_args['y_range'] = np.array([np.min(layout[:,1]), np.max(layout[:,1])])*[0.90, 1.10]
    p = figure(**fig_args)
    p.axis.visible = True
    p.xgrid.visible = p.ygrid.visible = True
    p.toolbar.logo = None

    ## Construct node data source
    node_data = { 'x' : layout[:,0], 'y' : layout[:,1], 'width' : np.log(v_sizes+1), 'n' : v_sizes }
    self.node_source = ColumnDataSource(node_data)
    self.node_glyphs = p.circle('x', 'y', size='width', color='#ff0000', alpha=1.0, source=self.node_source)
    
    ## Create node size slider (optional)
    self.node_size_slider = Slider(start=0.01, end=0.75, step=0.01, value=0.05)
    self.node_size_slider.js_link('value', self.node_glyphs.glyph, 'radius')
    
    ## Construct edge data source
    r, c = np.where(G.A)
    R, C = r[r < c], c[r < c]
    ES, EX, EY = np.ravel(G[R,C]), list(np.c_[layout[R,0], layout[C,0]]), list(np.c_[layout[R,1], layout[C,1]])
    edge_data = dict(xs=EX, ys=EY, color=np.repeat('black', len(EX)), line_width=(ES / np.max(ES))*2, n=ES)
    self.edge_source = ColumnDataSource(data=edge_data)
    self.edge_glyphs = p.multi_line(xs="xs", ys="ys", color='color', line_width='line_width', alpha=0.80, source=self.edge_source)

    assert len(check_integrity([p]).error) == 0
    self.p = p

  def plot(self, notebook: bool = True):
    if notebook: 
      output_notebook(verbose=False, hide_banner=True)
    show(self.p)

  def plot_dynamic(self, f: Callable, notebook: bool = True):
    # embed_p = figure(tools=TOOLS, width=500, height=500, title=None)
    # embed_p.circle('x', 'y', source=source)
    if notebook: 
      output_notebook(verbose=False, hide_banner=True)
    s = slice(None)
    # self.node_source.patch({ 'x' : [(s, X[:,0])], 'y' : [(s, X[:,1])] })
    def update():
      print("hello")
      # s = slice(None)
      X = f()
      self.node_source.patch({ 'x' : [(s, X[:,0])], 'y' : [(s, X[:,1])] })
    n_frames = 10  
    curdoc().add_periodic_callback(update, 1000/n_frames) # 1
    show(self.p)
    # self.p = gridplot([[self.p, embed_p]])
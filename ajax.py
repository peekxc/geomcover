# bokeh serve --show force_server.py
from typing import * 
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from bokeh.models import AjaxDataSource, CustomJS, ColumnDataSource, Range1d
from bokeh.plotting import figure, show
import networkx as nx
from fr_nx import ForceLayout
# from fastapi.encoders import jsonable_encoder
# from ph_force import EMST, subgraph_cut, components_cut, tree_cutset, mst_cutset

from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Rect, Slider
from bokeh.io import curdoc
from bokeh.layouts import gridplot, row, column

from sklearn.neighbors import radius_neighbors_graph
from scipy.spatial.distance import pdist, cdist



# G = nx.watts_strogatz_graph(n=150, k=5, p=0.14)
X = np.random.uniform(size=(150, 2))
G, T = EMST(X)
G = nx.Graph(radius_neighbors_graph(X, radius = np.quantile(pdist(X), 0.05)))
# T = EMST


fg = ForceLayout(G, pos = X)


E = T.edges(data=True)
W = np.array([a['weight'] for (u,v,a) in E])



di = np.ptp(X, axis=0)
l, r = np.min(X[:,0])-2*di[0], np.max(X[:,0])+2*di[0]
b, t = np.min(X[:,1])-2*di[1], np.max(X[:,1])+2*di[1]

p = figure(
	height=800, width=900, background_fill_color="white",
	title="Force directed layout", 
	#output_backend="webgl",
	x_range=(l, r), y_range=(b, t), 
	toolbar_location="above", 
	tools = "lasso_select, pan, poly_select, tap, wheel_zoom, reset, point_draw"
)
# configure so that no drag tools are active
p.toolbar.active_drag = None
p.toolbar.active_scroll = None
p.toolbar.active_tap = None
p.toolbar.active_inspect = None

## Create node data source 
node_data = {
	'x': list(fg._pos[:,0]),
	'y': list(fg._pos[:,1])
}
node_source = ColumnDataSource(data=node_data)


## Add edges via multiline 
def compute_edge_coords(fg):
	xs = list([fg._pos[u,0],fg._pos[v,0]]  for (u,v) in fg.graph.edges)
	ys = list([fg._pos[u,1],fg._pos[v,1]]  for (u,v) in fg.graph.edges)
	return({ 'xs': xs, 'ys': ys })

## Create edge data source 
edge_data = compute_edge_coords(fg)
edge_data['color'] = list(np.repeat('#ff0000', len(edge_data['xs'])))
edge_source = ColumnDataSource(data=edge_data)

p.multi_line('xs', 'ys', alpha=0.6, color='color', source=edge_source, line_width=3)
node_glyphs = p.circle(x='x', y='y', color="blue", alpha=1.0, source=node_source, radius=0.015)


node_size_slider = Slider(start=0.01, end=0.75, step=0.01, value=0.05)
node_size_slider.js_link('value', node_glyphs.glyph, 'radius')

## Persistence intervals
import networkx as nx
from ph_force import EMST
g, emst = EMST(X)
w = np.array([a['weight'] for (u,v,a) in emst.edges(data=True)])

p2 = figure(
	height=800, width=190, background_fill_color="white",
	title="Persistence intervals", 
	y_range=Range1d(0, len(w), bounds="auto"),
	x_range=Range1d(0, np.max(w), bounds="auto"),
	tools="box_select, box_zoom, tap, wheel_zoom, reset", 
	toolbar_sticky=False, 
	toolbar_location="above"
)
p2.yaxis.visible = False
p2.toolbar.active_drag = None # make no toolbar active

pos_ind = np.argsort(w)
w = w[pos_ind]
ind = pos_ind.copy()
for i, c in enumerate(pos_ind): ind[c] = i # invert pos_ind


# source = ColumnDataSource(dict(x=x, y=y, w=w, h=h))
bar_height = 1
bar_x = w/2.0
bar_y = bar_height*np.arange(len(w))+1 - (bar_height/2.0)
bar_w = w
bar_h = np.repeat(bar_height, len(w))
bar_source = ColumnDataSource(dict(x=bar_x, y=bar_y, w=bar_w, h=bar_h, color=np.repeat("#cab2d6", len(w))))
bar_glyph = Rect(x="x", y="y", width="w", height="h", angle=0.0, fill_color="color")
p2.add_glyph(bar_source, bar_glyph)

bar_source.selected.js_on_change("indices", CustomJS(args=dict(bar_source=bar_source, edge_source=edge_source), code="""
	const inds = cb_obj.indices;
	const bar_data = bar_source.data;
	console.log(inds)
	let c = 0; 
	for (let i = 0; i < bar_data['color'].length; i++){
		bar_data['color'][i] = "#cab2d6"
	}
	for (let i = 0; i < inds.length; i++) {
		bar_data['color'][inds[i]] = "#ff0000"
	} 
	bar_source.change.emit()
"""))

E = np.array(emst.edges())[pos_ind]

# def log_bar_selected(attr, old, new):
# 	print("bar selected!")
# 	print(attr)
# 	print(old)
# 	print(new)
# 	s = slice(None)
	
# 	ne = len(edge_source.data['color'])
# 	ec = np.repeat("#ff0000", ne)
# 	ec[old] = "#ff0000"
# 	ec[new] = "#cab2d6"
# 	print(ec)
# 	edge_source.patch({ 'color' : [(s, list(ec))] })
# bar_source.selected.on_change('indices', log_bar_selected)

tree_edges = list(T.edges(data=True))

TOOLTIPS = [
	("index", "$index"),
	("size", "@n"),
]

def highlight_edge_components(attr, old, new): 
	print("bar selected!")
	print(attr)
	print(old)
	print(new)
	s = slice(None)
	# mst_cutset(T, (98, 120), 0.0717)[1]

bar_source.selected.on_change('indices', highlight_edge_components)

# Add slider 
bar_slider = Slider(title='Threshold', start=0, end=np.max(w), step=np.max(w)/50, value=0.0, width=200)

## Add vertical bar that changes with the slider
threshold_callback = CustomJS(args=dict(source=bar_source), code="""
	const data = source.data;
	const f = cb_obj.value 
	const w = data['w']
	console.log(data)
	
	for (let i = 0; i < w.length; i++) {
		if (w[i] < f){
			data['color'][i] = 'gray'
		} else {
			data['color'][i] = '#cab2d6'
		}
	}
	source.change.emit();
""")
bar_slider.js_on_change('value', threshold_callback)



def update():
	# print("hello")
	fg.step_force()
	ec = compute_edge_coords(fg)
	s = slice(None)
	edge_source.patch({ 'xs' : [(s, ec['xs'])], 'ys' : [(s, ec['ys'])] })
	node_source.patch({ 'x' : [(s, fg._pos[:,0])], 'y' : [(s, fg._pos[:,1])] })


## Refresh at n_frames per second
n_frames = 30  
curdoc().add_periodic_callback(update, 1000/n_frames) # 1000/60
curdoc().add_root(row(column(p, node_size_slider), column(p2, bar_slider)))

# app = FastAPI()

# origins = ["*"] 
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# x = list(np.arange(0, 6, 0.1))
# y = list(np.sin(x) + np.random.random(len(x)))


# @app.get('/data')
# async def data():
# 	x.append(x[-1]+0.1)
# 	y.append(np.sin(x[-1])+np.random.random())
# 	return jsonable_encoder({'x': x, 'y': y})

# from fastapi.encoders import jsonable_encoder
# @app.post('/node_coordinates')
# async def edge_coordinates():
# 	fg.step_force()
# 	return jsonable_encoder({ 'x': list(fg._pos[:,0]), 'y': list(fg._pos[:,1]) })

# xy_identity = CustomJS(code="""return cb_data.response""")




# p = figure(
# 	height=300, width=800, background_fill_color="white",
# 	title="Streaming Noisy sin(x) via Ajax", 
# 	# output_backend="webgl"
# )






# p.circle(fg._pos[:,0], fg._pos[:,0], radius = 0.01)



# p.circle(x='x', y='y', source=node_source)

# Alternatively, if the REST API returns a different format, a CustomJS callback can be provided to convert the REST response into Bokeh format, via the adapter property of this data source.
# adapter = CustomJS(code="""
#     console.log(cb_data)
# 		return(cb_data.response)
# """)
# ajax_node_source = AjaxDataSource(
# 	data_url='http://localhost:8001/data',
# 	polling_interval=2000, 
# 	method='POST',
# 	adapter=adapter
# )
# node_data = {
# 	'x': list(fg._pos[:,0]),
# 	'y': list(fg._pos[:,1])
# }
# node_source = ColumnDataSource(data=node_data)
# p.circle(x='x', y='y', source=ajax_node_source)


# # from fastapi.encoders import jsonable_encoder
# @app.post('/data')
# def node_coordinates():
# 	# slice_x_patch = ( slice(None), list(fg._pos[:,0]) )
# 	# node_source.patch({ 'x' : [slice_x_patch] })
# 	return jsonable_encoder({ 'x': list(fg._pos[:,0]), 'y': list(fg._pos[:,1]) })

# @app.post('/data/again')
# def edge_coordinates():
# 	edge_json = jsonable_encoder(compute_edge_coords(fg))
# 	fg.step_force()
# 	return(edge_json)


# ajax_node_source2 = AjaxDataSource(
# 	data_url='http://localhost:8001/data/again',
# 	polling_interval=2000, 
# 	method='POST',
# 	adapter=adapter
# )
# p.multi_line(xs='xs', ys='ys', source=ajax_node_source2, color='red', alpha=0.35)

	# { 'ys' : [([0, i], [new_y])]}
	# edge_source.data = compute_edge_coords(fg)
	# edge_source.trigger('change')

	# return jsonable_encoder({ 'x': list(fg._pos[:,0]), 'y': list(fg._pos[:,1]) })


# Conclusion: just use https://www.youtube.com/watch?v=WgyTSsVtc7o bokeh serve


# @app.post('/edge_coordinates')
# def edge_coordinates():
# 	console.log("no idea")
# 	return jsonable_encoder({ 'x': list(fg._pos[:,0]), 'y': list(fg._pos[:,1]) })

# p.circle('x', 'y', source=source)
# p.x_range.follow = "end"
# p.x_range.follow_interval = 10




# if __name__ == "__main__":
# 	import uvicorn
# 	uvicorn.run(app,host="localhost",port=8001)
# app.run(port=5050)
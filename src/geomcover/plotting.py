import numpy as np 


def plot_cover(H, pos, offset: float = 0.15, **kwargs):
	from bokeh.plotting import figure
	from scipy.spatial import ConvexHull
	from shapely.geometry import Polygon
	from shapely import offset_curve
	h,w = kwargs.pop("height", 150), kwargs.pop("width", 150)
	p = figure(height=h, width=w, **kwargs)
	p.toolbar_location = None
	p.xaxis.visible = False 
	p.yaxis.visible = False
	p.grid.visible = False
	for he in H:
		if len(he) == 1:
			p.circle(*pos[he[0]], radius=offset, fill_alpha=0, line_color='blue', line_width=2)
		elif len(he) == 2: 
			s = pos[np.append(he, he[0])]
			oc = offset_curve(Polygon(s), offset)
			p.line(*oc.xy, line_width=2, line_color='orange')
		else:
			s = pos[np.append(he, he[0])]
			hull = ConvexHull(s)
			oc = offset_curve(Polygon(s[hull.vertices]), offset)
			p.line(*oc.xy, line_width=2, line_color='red')
	p.scatter(*pos.T, color='black', size=7)
	return p


def plot_tangent_bundle(TM, c: float = 1.0, data: bool = False, **kwargs):
	"""Plots a 1D or 2D tangent bundle for 2D data."""
	from bokeh.plotting import figure
	xs = [np.ravel((pt[0]-c*v[0,0], pt[0]+c*v[0,0])) for pt,v in TM]
	ys = [np.ravel((pt[1]-c*v[1,0], pt[1]+c*v[1,0])) for pt,v in TM]
	S = np.array([pt for pt,v in TM])
	assert S.ndim == 2 and S.shape[1] == 2, "Can only plot 2D points."
	if data: 
		return (S[:,0], S[:,1]), (xs, ys)
	else: 
		p = figure(**kwargs)
		assert isinstance(p, figure), "Passed plot object 'p' must be a bokeh figure"
		p.multi_line(xs,ys, color='red', line_width=1.20, line_alpha=0.50)
		p.scatter(*S.T, color='gray', size=3.2, line_width=0.5, alpha=0.50, line_color='gray')
		return p

def plot_nerve(M, X, **kwargs):
	"""Plots the nerve of a given cover."""
	from bokeh.plotting import figure
	A = M.tocoo()
	# st = SimplexTree(zip(A.row, A.col))
	# st.expand(5)
	xs = list(zip(X[A.row,0], X[A.col,0]))
	ys = list(zip(X[A.row,1], X[A.col,1]))
	p = figure(**kwargs)
	p.multi_line(xs,ys, color='black', line_width=0.80, line_alpha=0.30)
	p.scatter(*X.T, color='red', size=3.4, line_width=0.5, line_color='gray')
	return p
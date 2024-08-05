
# %% Imports
import numpy as np
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
output_notebook()

# %%
from scipy.stats import norm
np.random.seed(1234)
dom = norm(loc=0, scale=1).rvs(size=500)
# dom = uniform(loc=0, scale=1).rvs(size=200)
# prop_con = max(norm.ppf(0.60), np.max(np.abs(dom)))
# qt = uniform.ppf(0.98)
qt = norm.ppf(0.96)
sample = np.where(np.abs(dom) <= qt, dom, np.nan)
sample = sample[~np.isnan(sample)]
prop_con = qt
theta = (sample / prop_con) * np.pi

X = np.c_[np.cos(theta), np.sin(theta)]
X *= np.random.uniform(size=len(X), low=0.90, high=1.10)[:,np.newaxis]

p = figure(width=350, height=350, match_aspect=True)
p.scatter(*X.T)
show(p)


from dreimac import CircularCoords
from map2color.color import map2hex, BokehColorPalette
viridis = BokehColorPalette().lookup('viridis')
palette = viridis # _lerp_palette(np.append(viridis, np.flip(viridis)), 256)

CC = CircularCoords(X, n_landmarks=400)
cc = CC.get_coordinates()

# mapped_theta = (theta + np.pi)
# p = figure(width=350, height=350, match_aspect=True, title="Inferred circular coordinate")
# q = figure(width=350, height=350, match_aspect=True, title="True circular coordinate")
# p.scatter(*X.T, color=map2hex(cc,palette), size=5)
# q.scatter(*X.T, color=map2hex(mapped_theta, palette),size=5)
# show(row(p, q))

cc_norm = cc / np.max(cc) # (np.pi*2)
# th_norm = theta / np.max(theta) #(np.pi*2)
th_norm = (theta + np.pi)/(2*np.pi)
# cc_norm = 1.0 - cc_norm

p = figure(width=350, height=350, match_aspect=True, title="Inferred circular coordinate")
q = figure(width=350, height=350, match_aspect=True, title="True circular coordinate")
p.scatter(*X.T, color=map2hex((cc_norm[np.argmin(th_norm)] - cc_norm) % 1.0,palette, low=0, high=1), size=5)
q.scatter(*X.T, color=map2hex(th_norm, palette, low=0, high=1),size=5)
show(row(p, q))

# %% Sw1Pers example
from pbsig.persistence import sw_parameters, sliding_window
from set_cover.covers import pca

np.random.seed(1236)
f = lambda t: np.cos(t) + np.cos(3*t)
SW = sliding_window(f, bounds=(0, 12*np.pi))
_, tau_opt = sw_parameters(bounds=(0, 12*np.pi), d=24, L=6)
N, M = 400, 40 # num of points, dimension 
X = SW(n=N, d=M, L=6) ## should be the perfect period
Y = X + np.random.uniform(size=X.shape, low=-0.50, high=0.50)
CC = CircularCoords(X, n_landmarks=400)
cc_truth = CC.get_coordinates() # perc=0.9

figs = []
emb = pca(Y)
for nl in [50,100,150,200,250,300,350,400]:
  CC = CircularCoords(Y, n_landmarks=nl)
  cc = CC.get_coordinates() # perc=0.9
  p = figure(width=150, height=150, match_aspect=True, title=f"SW1Pers CC ({nl})")
  p.scatter(*emb.T, color=map2hex(cc, palette), size=3.0)
  p.xaxis.visible = False 
  p.yaxis.visible = False
  p.toolbar_location = None
  p.title.align = 'center'
  figs.append(p)

p = figure(width=300, height=300, match_aspect=True, title="SW1Pers CC (ground truth)")
p.scatter(*emb.T, color=map2hex(cc, palette), size=5.0)
p.xaxis.visible = False 
p.yaxis.visible = False
p.toolbar_location = None
show(row(p, column(row(figs[:4]), row(figs[4:]))))

# %% Augment Dreimac 





# p = figure(width=350, height=350, match_aspect=True, title="Inferred circular coordinate")
# p.scatter(*pca(X).T, color=map2hex(cc, palette), size=5)
# show(p)


# %% Plot periodic function

w = tau_opt * 24

p = figure(width=450, height=225)
p.line(dom, f(dom))
p.rect(x=w/2, y=0, width=w, height=4, fill_alpha=0.20, fill_color='gray', line_alpha=0, line_width=0)
show(p)

# %% Make a slidding window embedding
from pbsig.linalg import pca
N, M = 150, 23 # num of points, dimension 
X_delay = SW(n=N, d=M, L=6) ## should be the perfect period



## Just need to 
# from scipy.optimize import minimize_scalar
# obj = lambda offset: np.linalg.norm(((cc_norm + offset) % 1.0) - th_norm)
# offset = minimize_scalar(obj, bounds=(0,1)).x




# %% Custom dreimac? 



# import pbsig.dsp 
# from pbsig.dsp import phase_align
# offset = phase_align(theta, cc, return_offset=True)

# dom = expon(0.0, scale=1/0.25).rvs(size=300)
# prop_con = max(expon.ppf(0.99), np.max(dom))
# theta = (dom / prop_con) * np.pi * 2.0

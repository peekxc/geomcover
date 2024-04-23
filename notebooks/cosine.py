import numpy as np 
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from landmark import landmarks
output_notebook()


theta = np.linspace(0, 12*np.pi, 150, endpoint=True)
y = np.cos(theta)

# %% Show cosine wave
p = figure(width=450, height=150)
p.line(theta, y, color='blue')
p.scatter(theta, y, color='blue', size=3)
show(p)

# %% 
from set_cover import wset_cover_LP



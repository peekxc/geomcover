import numpy as np
import scipy.io
mat = scipy.io.loadmat('/Users/mpiekenbrock/Downloads/cardiac32ch_b1.mat')

from itertools import product
import matplotlib.pyplot as plt 
f, axarr = plt.subplots(4,8, figsize=(12,8)) 

for c, (i,j) in enumerate(product(range(4), range(8))):
  mri_data = mat['data'][:,:,0,c]
  mri_data = mri_data.astype(float)
  axarr[i,j].imshow(mri_data, cmap='gray')


base_img = np.zeros(shape=mat['data'][:,:,0,0].shape)
for c in range(32):
  mri_data = mat['data'][:,:,0,c]
  #mri_data = mri_data.astype(float)
  #mri_data = abs(np.real(mri_data)) + abs(np.imag(mri_data))
  mri_data = np.sqrt(np.real(mri_data)**2 + np.imag(mri_data)**2)
  base_img += mri_data
plt.imshow(base_img, cmap='gray')


all_mri_images = []
for c in range(23):
  base_img = np.zeros(shape=mat['data'][:,:,0,0].shape)
  for j in range(32):
    mri_data = mat['data'][:,:,c,j]
    mri_data = np.sqrt(np.real(mri_data)**2 + np.imag(mri_data)**2)
    base_img += mri_data
  all_mri_images.append(base_img.flatten())
  # axarr[c].imshow(base_img, cmap='gray')

import matplotlib.pyplot as plt 
# f, axarr = plt.subplots(1,23, figsize=(58,8), dpi=250) 
fig, ax = plt.subplots()
for i in range(len(all_mri_images)):
  ax.cla()
  ax.imshow(all_mri_images[i].reshape(base_img.shape))
  ax.set_title(f"frame {i}")
  plt.pause(0.1)
  plt.show()




from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots()
im = ax.imshow(all_mri_images[0].reshape(base_img.shape))

def update(frame):
  #ax.imshow(all_mri_images[frame].reshape(base_img.shape))
  print(frame)
  im.set_array(all_mri_images[int(frame)].reshape(base_img.shape))
  return [im]

ani = FuncAnimation(fig, update, frames=range(len(all_mri_images)), interval=500)
#ani.save('mri_anim.mp4', fps=5, extra_args=['-vcodec', 'libx264'])
ani.save('mri_anim.gif', fps=5, writer='imagemagick')
plt.show()
print('Done!')

from scipy.spatial.distance import pdist, cdist
Z = np.vstack(all_mri_images)

from tallem.dimred import cmds
X = cmds(pdist(Z)**2, 2)
plt.scatter(*X.T)


from tallem.circular_coordinates import CircularCoords
cc = CircularCoords(X, X.shape[0])
cw = cc.get_coordinates()

plt.scatter(*X.T, c=cw, s=45.5)

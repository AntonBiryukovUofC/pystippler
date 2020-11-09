import sys

from fast_gpu_voronoi import Instance
from fast_gpu_voronoi.jfa import JFA, JFA_mod, JFA_star
from fast_gpu_voronoi.debug import save
import numpy as np

npts = 10
size_xy = 50
random_pts = np.random.randint(2, size_xy-2, size=(npts, 2))
arr = [JFA, JFA_mod, JFA_star]

I = Instance(alg=arr[2], x=size_xy, y=size_xy, pts=random_pts)
I.run()
print(I.M.shape)  # (50, 50, 1)
save(I.M, I.x, I.y, force=True)  # __1_debug.png

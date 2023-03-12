# Trash uScope image filtering tests
# Imaad Syed
# 3/12/2023

import numpy as np
import cv2 as cv
import time as t
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

frames = 300

# video load
vid = cv.VideoCapture("vid.mov")

# storage for video
f = np.ones((722, 678, frames), dtype=np.uint8)

b = 0

# store frames in array of matrices

while(vid.isOpened()):
  a, fl = vid.read()
  g = cv.cvtColor(fl, cv.COLOR_BGR2GRAY)
  f[:, :, b] = g
  b = b + 1
  if(b >= 300):
      break

# z = x^2 + y^2 <--- should probably be changed to ellipse due to rectangular image and discrete Fourier frequency scaling
# for a high pass, we only take z > l

l = 1500

f_s = np.ones((722, 678));

# filter generation
for a in range(1,722):
  for b in range(1,678):
    if (( (a - 361) * (a - 361) ) + ((b - (678/2)) * (b - (678/2)))) < l:
      f_s[a,b] = 0

# output frame storage
f2 = np.ones((722, 678)).astype(np.int64)

z = []

# have to rescale image accordingly, as the camera goes down, the image zooms, which is causing problems with coordinate matching

#filter images and show
for i in range(254, 278):
  I_FFT = np.fft.fft2(f[:,:,i])
  I_FFT = np.fft.fftshift(I_FFT) * f_s
  img = np.fft.ifft2(I_FFT);
  comp = np.concatenate((abs(img).astype(np.uint8), f[:,:,i]), axis=0)
  cv.imshow("filtered image and original", comp)
  print("Frame #%d\n" % i)
  cv.waitKey(0)

  # up to 253
  # down: 254 to 278

  if i == 254:
    z = 0
  else:
    z = z + 1

  f2 = f2 + (z * ((2 * abs(img)) > 30))
  cv.imshow("layer", 255 * ((2 * abs(img)) > 30).astype(np.uint8))
  cv.waitKey(0)

# this has problems, will fix
cv.imshow("height map", f2 / ())
cv.waitKey(0)

xp, yp = np.mgrid[0:f2.shape[0], 0:f2.shape[1]]

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xp, yp, -1 * f2)
plt.show()
cv.waitKey(0)

vid.release()
cv.destroyAllWindows()

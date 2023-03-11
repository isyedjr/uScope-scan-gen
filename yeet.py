# Trash uScope image filtering tests
# Imaad Syed
# 3/10/2023

import numpy as np
import cv2 as cv
import time as t

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

l = 250

f_s = np.ones((722, 678));

# filter generation
for a in range(1,722):
  for b in range(1,678):
    if (( (a - 361) * (a - 361) ) + ((b - (678/2)) * (b - (678/2)))) < l:
      f_s[a,b] = 0

# output frame storage
f2 = np.ones((722, 678, (278 - 234) + 1))

#filter images and show
for i in range(234, 278):
  I_FFT = np.fft.fft2(f[:,:,i])
  I_FFT = np.fft.fftshift(I_FFT) * f_s
  img = np.fft.ifft2(I_FFT);
  comp = np.concatenate((abs(img).astype(np.uint8), f[:,:,i]), axis=0)
  cv.imshow("filtered image and original", comp)
  cv.waitKey(0)
  f2[:,:,i-234] = abs(img)

cv.waitKey(0)

vid.release()
cv.destroyAllWindows()

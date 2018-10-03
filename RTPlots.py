import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation


import functools
import io
import sys

## fixing numpy bug
#
genfromtxt_old = np.genfromtxt
@functools.wraps(genfromtxt_old)
def genfromtxt_py3_fixed(f, encoding="utf-8", *args, **kwargs):
  if isinstance(f, io.TextIOBase):
    if hasattr(f, "buffer") and hasattr(f.buffer, "raw") and \
    isinstance(f.buffer.raw, io.FileIO):
      # Best case: get underlying FileIO stream (binary!) and use that
      fb = f.buffer.raw
      # Reset cursor on the underlying object to match that on wrapper
      fb.seek(f.tell())
      result = genfromtxt_old(fb, *args, **kwargs)
      # Reset cursor on wrapper to match that of the underlying object
      f.seek(fb.tell())
    else:
      # Not very good but works: Put entire contents into BytesIO object,
      # otherwise same ideas as above
      old_cursor_pos = f.tell()
      fb = io.BytesIO(bytes(f.read(), encoding=encoding))
      result = genfromtxt_old(fb, *args, **kwargs)
      f.seek(old_cursor_pos + fb.tell())
  else:
    result = genfromtxt_old(f, *args, **kwargs)
  return result

if sys.version_info >= (3,):
  np.genfromtxt = genfromtxt_py3_fixed
#
##


path = '/home/francesco/PycharmProjects/SemesterProject/'

# Bleien position
B_long = 8.111203
B_lat = 47.33982
B_alt = 540.0

# Airport position
Z_long = 8.564572
Z_lat = 47.451542
Z_alt = 432.0


# Telescope params
Tazi = 309.9
Tele = 59.1





# 3D real time plot of planes
fig = plt.figure(figsize=(12,8))
def animate(i):
    # Upload Data
    f = open(path + 'dataOUT.csv', 'r')
    D = np.genfromtxt(f, delimiter=",")
    f.close()

    squawk = D[1:, 2]
    track = D[1:, 10]
    longi = D[1:, 5]  # [deg]
    alti = D[1:, 8]  # [F] ??
    state = D[1:, 9]
    lati = D[1:, 4]  # [deg]
    speed = D[1:, 11]  # [mph]
    dkm = 40000.0 / 360.0  # 111.1 km/degree
    dx = (longi - B_long) * dkm
    dy = (lati - B_lat) * dkm
    dh = (alti / 3.2808 - B_alt) / 1000
    ds = np.sqrt(dx ** 2 + dy ** 2)
    R = np.sqrt(ds ** 2 + dh ** 2)
    # Radial distance
    normR = R / np.max(R)
    c = 0.5 * np.pi * normR  # ? format of squawk?
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    bx = [B_long, B_long + 1]
    by = [B_lat, B_lat + 1]
    bz = [B_alt / 1000.0, 14]
    a1 = fig.add_subplot(111, projection='3d')
    a1.scatter(longi, lati, alti/3.2808/1000.0, c=c, marker='.', s=20/np.sqrt(normR), cmap='autumn', norm=norm)
    a1.scatter(B_long, B_lat, B_alt/1000.0, c='g', marker='*', s=200)
    a1.scatter(Z_long, Z_lat, Z_alt/1000.0, c='b', marker='o', s=100)
    a1.plot(bx, by, bz, color='g')
    a1.set_xlim(7, 10)
    a1.set_ylim(45, 49)
    a1.set_zlim(0, 16)
    a1.set_xlabel('Longitude [deg]', size=15)
    a1.set_ylabel('Latitude [deg]', size=15)
    a1.set_zlabel('Altitude[km]', size=15)
    a1.set_title('3D position airplanes', size=20)
#plt.savefig(path+'3D.png')
ani = animation.FuncAnimation(fig, animate, interval= 1000)
plt.show()


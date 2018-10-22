import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
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


path = '/home/francesco/PycharmProjects/SemesterProject/data/'
dkm =111.1
img_path = '/home/francesco/PycharmProjects/SemesterProject/figures/'

class Telescope:
  def __init__(self, long, lat, alt, azi, ele):
    self.long = long
    self.lat = lat
    self.alt = alt
    self.azi = azi*np.pi/180
    self.ele = ele

T = Telescope(8.111203, 47.33982, 540.0, 309.9, 90)


# Airport position
Z_long = 8.564572
Z_lat = 47.451542
Z_alt = 432.0

# Upload Data
f = open(path+'dataOUT.csv', 'r')
D = np.genfromtxt(f, delimiter=",", usecols=np.arange(0,18))
f.close()
lati =D[1:,4]
longi=D[1:,5]
alti= D[1:,8]


fig1 = plt.figure(figsize=(12,8))
a1 = fig1.add_subplot(111, projection='3d')
a1.scatter(lati, longi, alti/3.2808/1000.0, marker='.', c='r',s=100)
a1.scatter(T.lat, T.long, T.alt/1000.0, c='g', marker='*', s=200)
a1.scatter(Z_lat, Z_long, Z_alt/1000.0, c='b', marker='o', s=100)
#a1.plot(px, py, pz, color='g')
z = np.arange(0, 20.2/dkm, 0.2/dkm)
theta = np.arange(0, 2 * np.pi + np.pi / 50, np.pi / 50)
pxx = np.arange(T.lat, T.lat+np.cos(T.azi), np.cos(T.azi)/100)
pyy = np.arange(T.long, T.long+np.sin(T.azi), np.sin(T.azi)/100)
i=0
for zval in z:
    x = zval*3/50* np.array([np.cos(q) for q in theta])
    y = zval*3/50* np.array([np.sin(q) for q in theta])
    a1.plot(pxx[i]+x, pyy[i]+y, dkm*zval, 'b-')
    i+=1
a1.set_ylim(7, 10)
a1.set_xlim(45, 49)
a1.set_zlim(0, 16)
a1.set_xlabel('Longitude [deg]', size=15)
a1.set_ylabel('Latitude [deg]', size=15)
a1.set_zlabel('Altitude[km]', size=15)
a1.set_title('3D position airplanes', size=20)
plt.savefig(img_path+'3D.png')
plt.show()

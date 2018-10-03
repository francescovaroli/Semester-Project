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
#
##


path = '/home/francesco/PycharmProjects/SemesterProject/data/'
img_path = '/home/francesco/PycharmProjects/SemesterProject/figures/'

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

# Upload Data
f = open(path+'data_15s.txt', 'r')
D = np.genfromtxt(f, delimiter=",")
f.close()

squawk = D[:, 1]
track  = D[:, 4]
longi  = D[:, 5] # [deg]
alti   = D[:, 6] # [F] ??
state  = D[:, 9]
lati   = D[:, 12] # [deg]
speed  = D[:, 14] # [mph]

dkm = 40000.0/360.0 # 111.1 km/degree
dx = (longi-B_long)*dkm
dy = (lati-B_lat)*dkm
dh = (alti/3.2808-B_alt)/1000
ds = np.sqrt(dx**2 + dy**2)
R = np.sqrt(ds**2 + dh**2)
# Radial distance
normR = R/np.max(R)
c = 10*np.pi*normR#? format of squawk?
norm = mpl.colors.Normalize(vmin=0, vmax=1)
bx = [B_long, B_long+1]
by = [B_lat, B_lat+1]
bz = [B_alt/1000.0,14]

# 2D plot of planes
plt.figure(figsize=(6, 6))
plt.scatter(longi, lati, c=c, marker='o', s=5/np.sqrt(normR), cmap='autumn', norm=norm)
plt.plot(bx, by, c='g', markersize=5)
plt.xlabel("Longitude [deg]")
plt.ylabel("Latitude [deg]")
plt.title('Airplane positions around Bleien')
plt.xlim(6, 10.5)
plt.ylim(45, 50)
plt.grid()
plt.savefig(img_path+'2D.png')



# 3D plot of planes
fig = plt.figure(figsize=(12,8))
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
plt.savefig(img_path+'3D.png')
plt.show()

# Polar plot
P_azi = 180.0 + np.arctan2(dx, dy)/np.pi*180.0
a2 = plt.subplot(111, projection='polar')
a2.scatter(P_azi, R,  c=c, marker='o', s=5/np.sqrt(normR), cmap='autumn', norm=norm)
a2.scatter(Tazi, 0, c='g', marker='*', s=200)
a2.plot([Tazi, Tazi], [0, 300], color='g')
a2.set_xlim(0, 2*np.pi)
a2.set_ylim(0, 300)
a2.set_title('Position in polar coordinates ')
plt.savefig(img_path+'Polar.png')
plt.show()

# 2D plot of elevation
tx = np.arange(0,20,1)*0+Tazi
ty = np.arange(45-20, 45+20, 2.0)
P_ele = np.arcsin(dh/R)/np.pi*180.0
a3 = plt.subplot()
a3.scatter(P_azi, P_ele, c=np.abs(P_ele-np.max(P_ele)), marker='o', cmap='autumn')
a3.scatter(Tazi, Tele, c='g', marker='*', s=200)
#a3.plot(tx, ty, 'o',color='r', markersize=5)
a3.set_xlim(0, 360)
a3.set_ylim(0.90)
a3.set_title('Elevation')
a3.grid()
a3.set_xlabel("Azimut [deg]")
a3.set_ylabel("Elevation [deg]")
plt.savefig(img_path+'Elevation.png')
plt.show()

#print(help(plt))
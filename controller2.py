import requests, json, datetime, time, csv
import numpy as np
import functools
import io
import sys
from matplotlib import pyplot as plt
import matplotlib.animation as animation

dkm = 40000.0 / 360.0
dt = 2
url = 'http://129.132.63.205:8080/aircraft.json'
path = '/home/francesco/PycharmProjects/SemesterProject/data/dataOUT.csv'


class Data:
    def __init__(self, urld, file_path):
        self.url = urld
        self.file_path = file_path

    def downlad(self):
        try:
            new_data = json.loads(requests.get(self.url).content)['aircraft']
            now = str(datetime.datetime.now())
            fields = ['time', 'hex', 'squawk', 'flight', 'lat', 'lon', 'nucp', 'seen_pos', 'altitude', 'vert_rate',
                      'track', 'speed', 'category', 'mlat', 'tisb', 'messages', 'seen', 'rssi', 'type']
            with open(self.file_path, 'w') as writeFile:
                writer = csv.DictWriter(writeFile, fieldnames=fields)
                writer.writeheader()
                for line in new_data:
                    if all(k in line for k in ('lat', 'lon', 'altitude')):
                        line['time'] = now
                        writer.writerow(line)

        except:
            print("ERROR downloading data")
            pass


class Telescope:

  def __init__(self, lat, long, alt, azi, ele, radius, height):
      self.lat = lat
      self.long = long
      self.y = long*dkm
      self.x = lat*dkm
      self.alt = alt
      self.azi = azi
      self.ele = ele
      self.radius = radius
      self.height = height #radius/height  of the beam cone
      self.dir = [[self.lat, (self.lat + np.cos(self.azi*np.pi/180))],[self.long, (self.long + np.sin(self.azi*np.pi/180))],
                  [self.alt, self.alt + (np.sin(self.ele*np.pi/180))*dkm]]

  def to_T_coord(self, p):
      p = np.array(p)
      dp = p - np.array((self.x, self.y, self.alt))
      dp = dp.reshape((3, 1))
      theta1 = self.azi*np.pi/180
      theta2 = (90-self.ele)*np.pi/180 ##may be wrong the elevation angle,
      c1, s1 = np.cos(theta1), np.sin(theta1)
      c2, s2 = np.cos(theta2), np.sin(theta2)
      R1 = np.matrix([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
      R2 = np.matrix([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
      Tr = R1*R2
      return Tr.transpose()*dp

  def D2T_coord(self, P):
      m = np.array([])
      for p in P:
          m = np.append(m, self.to_T_coord(p))
      return m.reshape((-1, 3))

  def check_aircraft(self, P):
      m = np.array([])
      P_T = self.D2T_coord(P)
      for p_t in P_T:
          dist_xy = np.sqrt(p_t[0]**2 + p_t[1]**2)
          m = np.append(m, dist_xy>(self.radius/self.height)*p_t[2])
      return m

  def move(self):
      #TODO: write output with new azi, ele to move the real telescope
      pass

class PlanesP:
    def __init__(self, path):
        self.airport = [47.451542, 8.564572, 0.432]
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
        f = open(path, 'r')
        self.data = np.genfromtxt(f, delimiter=",", usecols=np.arange(0,18))
        f.close()
        self.hex = self.data[1:, 1]
        self.lat = self.data[1:, 4]
        self.long = self.data[1:, 5]
        self.alt = self.data[1:, 8]/3.2808/1000.0
        self.pos = np.array([[self.data[i, 4]*dkm, self.data[i, 5]*dkm, self.data[i, 8]/3.2808/1000.0] for i in range(1, self.lat.size+1)])#
        self.pos_t = T.D2T_coord(self.pos)
        self.R = np.sqrt((T.x-self.pos[:, 0])**2 + (T.y-self.pos[:, 1])**2)
        self.R_t = np.sqrt(self.pos_t[:, 0]**2 + self.pos_t[:, 1]**2)
        self.azi = 180+np.arctan2((self.lat-T.lat), (self.long-T.long))/np.pi*180.0
        d = self.pos-np.array([T.x, T.y, T.alt])
        self.ele = np.arcsin((self.alt-T.alt)/np.linalg.norm(d, axis=1))/np.pi*180

    def plot3D(self):
        fig = plt.figure(figsize=(10, 10))
        a = fig.add_subplot(111, projection='3d')
        s = a.scatter(self.lat, self.long, self.alt, c=self.R_t, marker='.', s=160, cmap='autumn') #norm=mpl.colors.Normalize(vmin=np.min(self.R), vmax=np.max(self.R))
        a.scatter(T.lat, T.long, T.alt, c='g', marker='*', s=200)
        a.scatter(self.airport[0], self.airport[1], self.airport[2], c='b', marker='o', s=100)
        a.plot(T.dir[0], T.dir[1], T.dir[2], color='g')
        a.set_ylim(7, 11)
        a.set_xlim(45, 49)
        a.set_zlim(0, 16)
        a.set_ylabel('Longitude [deg]', size=15)
        a.set_xlabel('Latitude [deg]', size=15)
        a.set_zlabel('Altitude[km]', size=15)
        a.set_title('3D position airplanes', size=20)
        a.invert_yaxis()
        cb = fig.colorbar(s)
        cb.set_label('Distance from beam', fontsize=14)
        plt.show()

    def plotPolar(self):
        fig = plt.figure(figsize=(9, 9))
        a = fig.add_subplot(111, projection='polar')
        a.set_xticklabels(['N', '', 'W', '', 'S', '','E', ''])
        l=a.scatter(self.azi*np.pi/180, self.R, c=self.R, marker='.', s=100, cmap='autumn')
        a.scatter(T.azi*np.pi/180, 0, c='g', marker='*', s=200)
        a.plot([T.azi*np.pi/180, T.azi*np.pi/180], [0, 300], color='g')
        a.set_xlim(0, 2 * np.pi)
        a.set_ylim(0, 300)
        cb = fig.colorbar(l)
        cb.set_label('Distance from beam', fontsize=14)
        a.set_title('Position in polar coordinates ')
        plt.show()

    def plotEle(self):
        a = plt.subplot()
        a.scatter(self.azi, self.ele, c=1/self.ele, marker='o', cmap='autumn')
        a.scatter(T.azi, T.ele, c='g', marker='*', s=200)
        a.set_xlim(0, 360)
        a.set_ylim(0,60)
        a.set_title('Elevation')
        a.grid()
        a.set_xlabel("Azimut [deg]")
        a.set_ylabel("Elevation [deg]")
        plt.show()

    def plotRT(self, tipe):
        fig = plt.figure(figsize=(12,8))
        ani = animation.FuncAnimation(fig, self.plotPolar, interval=4000) #functools.partial(chose_plot, tipe)
        plt.show(ani)


class Planes():

    def __init__(self, path):
        self.airport = [47.451542, 8.564572, 0.432]
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
        f = open(path, 'r')

        self.data = np.genfromtxt(f, delimiter=",", usecols=np.arange(0,18))
        f.close()
        self.hex = str(self.data[1:, 1])
        self.lat = self.data[1:, 4]
        self.long = self.data[1:, 5]
        self.alt = self.data[1:, 8]/3.2808/1000.0
        self.speed = self.data[1:, 11]*1.609344/3600.0  # [mph] --> [km/s]
        self.dir = np.array([self.data[1:, 10], self.data[1:, 9]*0.3048/60.0])  # vert_rate [ft/min]->[km/s]
        self.pos = np.array([[self.data[i, 4]*dkm, self.data[i, 5]*dkm, self.data[i, 8]/3.2808/1000.0] for i in range(1, self.lat.size+1)])
        # self.prev_speed = self.speed
        self.mod = 0

        # need a Telescope named T to be existing
        # self.pos_t = T.D2T_coord(self.pos)
        # self.R = np.linalg.norm(self.pos-np.array([T.x, T.y, T.alt]), axis=1)  # absolute distance
        # self.R_b = np.sqrt(self.pos_t[:, 0]**2 + self.pos_t[:, 1]**2)  # distance from beam

        # self.azi = 180+np.arctan2((self.lat-T.lat), (self.long-T.long))/np.pi*180.0
        # self.ele = np.arcsin((self.alt-T.alt)/self.R)/np.pi*180


T = Telescope(47.33982, 8.111203, 540.0/1000, 270, 90, 3, 5)  # position initialized
data = Data(url, path)
first = 1

while True:
    data.downlad()
    P = Planes(path)
    if first:
        Ptemp = P
    for k in Ptemp.hex if k not in P.hex

    tel_pos = MPC(T, P, dt)

    T = Telescope(47.33982, 8.111203, 540.0/1000, tel_pos[0], tel_pos[1], 3, 5)
    T.move()

    time.sleep(dt)

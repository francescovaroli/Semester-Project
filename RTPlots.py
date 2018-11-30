import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import json, requests, datetime, csv

dkm = 40000.0 / 360.0  # 111.1 km/degree
path = '/home/francesco/PycharmProjects/SemesterProject/data/dataOUT.csv'
url = 'http://129.132.63.205:8080/aircraft.json'

class Plots:

    def plot3D(self, lati, longi, alti, R):
        fig.clear()
        a1 = fig.add_subplot(111, projection='3d')
        c =  R-np.min(R)
        norm = mpl.colors.Normalize(vmin=0, vmax=np.max(R)-np.min(R))
        s = a1.scatter(lati, longi, alti, c=c, marker='.', s=180, cmap='autumn', norm=norm)
        a1.scatter(T.lat, T.long, T.alt, c='g', marker='*', s=200)
        a1.scatter(z_lat, z_long, z_alt, c='b', marker='o', s=100)
        z = np.arange(0, 20.2 / dkm, 0.2 / dkm)
        theta = np.arange(0, 2 * np.pi + np.pi / 50, np.pi / 50)
        pxx = np.arange(T.lat, T.lat + np.cos(T.azi), np.cos(T.azi) / 100)
        pyy = np.arange(T.long, T.long + np.sin(T.azi), np.sin(T.azi) / 100)
        i = 0
        for zval in z:
            while i < len(pxx):
                x = zval * 3 / 50 * np.array([np.cos(q) for q in theta])
                y = zval * 3 / 50 * np.array([np.sin(q) for q in theta])
                a1.plot(pxx[i] + x, pyy[i] + y, dkm * zval, 'b-')
                i += 1
        a1.plot(T.dir[0], T.dir[1], T.dir[2], color='g')
        a1.set_ylim(7.5, 9.5)
        a1.set_xlim(45.5, 48.5)
        a1.set_zlim(0, 16)
        a1.set_xlabel('Longitude [deg]', size=15)
        a1.set_ylabel('Latitude [deg]', size=15)
        a1.set_zlabel('Altitude[km]', size=15)
        a1.set_title('3D position airplanes', size=20)
        cb = fig.colorbar(s)
        cb.set_label('Distance from beam [km]', fontsize=14)

    def plotPolar(self, azi, R):
        fig = plt.figure(figsize=(9, 9))
        a = fig.add_subplot(111, projection='polar')
        a.set_xticklabels(['N', '', 'W', '', 'S', '','E', ''])
        l=a.scatter(azi*np.pi/180, R, c=R, marker='.', s=100, cmap='autumn')
        a.scatter(T.azi*np.pi/180, 0, c='g', marker='*', s=200)
        a.plot([T.azi*np.pi/180, T.azi*np.pi/180], [0, 300], color='g')
        a.set_xlim(0, 2 * np.pi)
        a.set_ylim(0, 300)
        cb = fig.colorbar(l)
        cb.set_label('Distance from beam', fontsize=14)
        a.set_title('Position in polar coordinates ')
        plt.show()

    def plotEle(self,azi, ele):
        a = plt.subplot()
        a.scatter(azi, ele, c=1/ele, marker='o', cmap='autumn')
        a.scatter(T.azi, T.ele, c='g', marker='*', s=200)
        a.set_xlim(0, 360)
        a.set_ylim(0,60)
        a.set_title('Elevation')
        a.grid()
        a.set_xlabel("Azimut [deg]")
        a.set_ylabel("Elevation [deg]")
        plt.show()


class Data:
    def __init__(self, urld, file_path):
        self.url = urld
        self.file_path = file_path

    def downlad(self):
        n = 0
        try:
            new_data = json.loads(requests.get(self.url).content)['aircraft']
            now = str(datetime.datetime.now())
            fields = ['time', 'hex', 'flight', 'lat', 'lon', 'altitude', 'vert_rate',
                      'track', 'speed', 'rssi', 'seen_pos']
            with open(self.file_path, 'w') as writeFile:
                writer = csv.DictWriter(writeFile, fieldnames=fields)
                # writer.writeheader()
                for line in new_data:
                    if all(t in line for t in ('lat', 'lon', 'altitude')):
                        n = n+1
                        for l in ('nucp', 'squawk', 'category', 'mlat', 'tisb', 'messages', 'seen', 'type'):
                            if l in line:
                                del line[l]
                        if 'flight' not in line:
                            line['flight'] = 'unknown'
                        if line['altitude'] == 'ground':
                            line['altitude'] = 0
                        # TODO: deal with missing speed & vert_rate (maybe in Plane)
                        # if 'speed' not in line:    better use previous values if available
                        #    line['speed'] = 0
                        # if 'vert_rate' not in line:
                        #    line['vert_rate'] = 0
                        line['time'] = now
                        writer.writerow(line)

        except ImportError:
            print("ERROR downloading data")

        return n


class Telescope:

    def __init__(self, azi, ele):
        self.lat = 47.33982
        self.long = 8.111203
        self.y = self.long*dkm
        self.x = self.lat*dkm
        self.alt = 540.0/1000
        self.azi = azi
        self.ele = ele
        self.max_speed = 0.23   # [deg/s]
        self.beam = 3.25*np.pi/180  # degrees of the beam cone
        self.dir = [[self.lat, (self.lat + np.cos(self.azi*np.pi/180))],
                    [self.long, (self.long + np.sin(self.azi*np.pi/180))],
                    [self.alt, self.alt + (np.sin(self.ele*np.pi/180))*dkm]]
        self.sky = np.zeros((60, 15))


    def to_t_coord(self, p):
        dp = p - np.array((self.x, self.y, self.alt))
        dp = dp.reshape((3, 1))
        theta1 = self.azi*np.pi/180
        theta2 = (90-self.ele)*np.pi/180  # may be wrong the elevation angle,
        c1, s1 = np.cos(theta1), np.sin(theta1)
        c2, s2 = np.cos(theta2), np.sin(theta2)
        r1 = np.matrix([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
        r2 = np.matrix([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
        tr = r1*r2
        return tr.transpose()*dp


# Airport position
z_long = 8.564572
z_lat = 47.451542
z_alt = 432.0/1000.0

T = Telescope(150, 13.5)

data = Data(url, path)

plots = Plots()

# 3D real time plot of planes
fig = plt.figure(figsize=(14,10))
def animate(i):
    # Upload Data
    data.downlad()
    f = open(path, 'r')
    planes = np.genfromtxt(f, delimiter=",", usecols=np.arange(3, 6))
    f.close()

    long = planes[:, 1]  # [deg]
    alt = planes[:, 2]/3.2808/1000.0  # [F]
    lat = planes[:, 0]  # [deg]
    azi = [180 + np.arctan2((lat[i] - T.lat), (long[i] - T.long)) / np.pi * 180.0 for i in range(0, len(lat))]
    d = np.array([lat*dkm-T.x, long*dkm-T.y, alt-T.alt])
    ele = [np.arcsin((alt[j] - T.alt) / np.linalg.norm(d, axis=1)) / np.pi * 180 for j in range(0, len(lat))]
    #R = np.sqrt((T.x - d[0, :]) ** 2 + (T.y - d[1, :]) ** 2 + ((T.alt-d[2,:])*dkm)**2)
    R = [np.linalg.norm(T.to_t_coord(planes[i, :])) for i in range(0, len(lat))]
    plots.plot3D(lat, long, alt, R)

ani = animation.FuncAnimation(fig, animate, interval= 40)
plt.show()


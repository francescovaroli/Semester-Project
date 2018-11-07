import requests, json, datetime, time, csv
import numpy as np
import functools
import io
import sys
from matplotlib import pyplot as plt

dkm = 40000.0 / 360.0
dt = 2
url = 'http://129.132.63.205:8080/aircraft.json'
path = '/home/francesco/PycharmProjects/SemesterProject/data/dataOUT.csv'
max_dist = 500


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

    def __init__(self, lat, long, alt, azi, ele, beam):
        self.lat = lat
        self.long = long
        self.y = long*dkm
        self.x = lat*dkm
        self.alt = alt
        self.azi = azi
        self.ele = ele
        self.beam = beam*np.pi/180  # degrees of the beam cone
        self.dir = [[self.lat, (self.lat + np.cos(self.azi*np.pi/180))],
                    [self.long, (self.long + np.sin(self.azi*np.pi/180))],
                    [self.alt, self.alt + (np.sin(self.ele*np.pi/180))*dkm]]
        self.sky = np.zeros((15, 60))

    def grid2pos(self, g):  # g = [e,a]  grid indices
        ele = 15 + 4.5*g[0]
        azi = 47 + 4.5*g[1]
        return [ele, azi]

    def pos2grid(self, ele, azi):
        a = np.arange(45 + 2, 314, 4.5)
        e = np.arange(13 + 2, 81, 4.5)
        e_idx = np.argmin(np.abs(ele-e))
        a_idx = np.argmin(np.abs(azi-a))
        return [e_idx, a_idx]

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

    def d2t_coord(self, pn):
        m = np.array([])
        for p in pn:
            m = np.append(m, self.to_t_coord(p.pos))
        return m.reshape((-1, 3))

    def check_aircraft(self, p):
        m = np.array([])
        p_t = self.d2t_coord(p)
        for p_t in p_t:
            dist_xy = np.sqrt(p_t[0]**2 + p_t[1]**2)
            m = np.append(m, dist_xy > (np.sin(self.beam))*p_t[2])
        return m

    def mpc(self, pl):
        pass
        # TODO: controller

    def move(self):
        # TODO: write output with new azi, ele to move the real telescope
        pass


class Plane:

    def __init__(self, path, i):
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

        ff = open(path, 'r')
        strings = np.genfromtxt(ff, delimiter=",", dtype=None, usecols=(1))
        ff.close()
        ff = open(path, 'r')
        numbers = np.genfromtxt(ff, delimiter=",", dtype=float, usecols=np.arange(3, 11))
        ff.close()
        self.hex = strings[i]
        self.lat = numbers[i, 0]
        self.long = numbers[i, 1]
        self.alt = numbers[i, 2]/3.2808/1000.0
        self.speed = numbers[i, 5]*1.8520/3600.0  # [knots] --> [km/s]  may be mph => *1.609344
        self.dir = numbers[i, 4]*np.pi/180
        self.vert_rate = numbers[i, 3]*0.3048/60.0/1000.0  # vert_rate [ft/min]->[km/s]
        self.rssi = numbers[i, 6]
        self.pos = np.array([self.lat*dkm, self.long*dkm, self.alt])
        self.R = np.linalg.norm(self.pos-np.array([47.33982*dkm, 8.1112*dkm, 0.540]), axis=0)  # absolute distance
        self.prev_speed = self.speed
        self.pr_vert_rate = self.vert_rate
        self.ele = np.arcsin((self.alt-0.54)/self.R)/np.pi*180.0
        self.azi = 180.0 + np.arctan2((self.pos[0]-T.x), (self.pos[1])-T.y)/np.pi*180.0
        self.last_seen = numbers[i, 7]
        self.mod = 0
        self.model_fo(self.last_seen)

    def model(self, t):   # base modelling, keeps previous speed, direction, vertical rate
        self.lat = self.lat + self.speed*t*np.cos(self.dir)/dkm
        self.long = self.long + self.speed*t*np.sin(self.dir)/dkm
        self.alt = self.alt + t*self.vert_rate
        self.pos = np.array([self.lat*dkm, self.long*dkm, self.alt])
        self.R = np.linalg.norm(self.pos-np.array([47.33982*dkm, 8.1112*dkm, 0.540]), axis=0)  # absolute distance
        self.mod = self.mod + 1
        return self

    def model_fo(self, t):  # modelling with constant acceleration ---> effective only whe we have already seen the plane
        dv = self.speed - self.prev_speed
        dvr = self.vert_rate - self.pr_vert_rate
        self.lat = self.lat + (self.speed + 0.5*dv)*np.cos(self.dir)*t/dkm
        self.long = self.long + (self.speed + 0.5*dv)*t*np.sin(self.dir)/dkm
        self.alt = self.alt + t*(self.vert_rate + 0.5*dvr)
        self.prev_speed = self.speed
        self.speed = self.speed + dv
        self.vert_rate = self.vert_rate + dvr
        self.pos = np.array([self.lat*dkm, self.long*dkm, self.alt])
        self.R = np.linalg.norm(self.pos-np.array([47.33982*dkm, 8.1112*dkm, 0.540]), axis=0)  # absolute distance
        self.ele = np.arcsin((self.alt-0.54)/self.R)/np.pi*180.0
        self.azi = 180.0 + np.arctan2((self.pos[0]-T.x), (self.pos[1])-T.y)/np.pi*180.0
        self.last_seen = 0
        self.mod = self.mod + 1
        return self

    def model_vr(self, t):  # modelling with constant acceleration
        speed_xy = self.speed*np.sin(np.arccos(self.vert_rate/self.speed))
        prev_speed_xy = self.prev_speed*np.sin(np.arccos(self.pr_vert_rate/self.prev_speed))
        dv = speed_xy - prev_speed_xy
        dvr = self.vert_rate - self.pr_vert_rate
        self.lat = self.lat + (speed_xy + 0.5*dv)*np.cos(self.dir)*t/dkm
        self.long = self.long + (speed_xy + 0.5*dv)*t*np.sin(self.dir)/dkm
        self.alt = self.alt + t*(self.vert_rate + 0.5*dvr)
        self.prev_speed = self.speed
        self.speed = self.speed + self.speed -self.prev_speed
        self.vert_rate = self.vert_rate + dvr
        self.pos = np.array([self.lat*dkm, self.long*dkm, self.alt])
        self.R = np.linalg.norm(self.pos-np.array([47.33982*dkm, 8.1112*dkm, 0.540]), axis=0)  # absolute distance
        self.ele = np.arcsin((self.alt-0.54)/self.R)/np.pi*180.0
        self.azi = 180.0 + np.arctan2((self.pos[0]-T.x), (self.pos[1])-T.y)/np.pi*180.0
        self.last_seen = 0
        self.mod = self.mod + 1
        return self

T = Telescope(47.33982, 8.111203, 540.0/1000, 270, 90, 3.25)  # position initialized
data = Data(url, path)
first = 1

n_pl0 = data.downlad()
planes = [Plane(path, i) for i in range(0, n_pl0)]
planes_0 = [Plane(path, i) for i in range(0, n_pl0)]
planes2 = [Plane(path, i) for i in range(0, n_pl0)]

for k in range(0, n_pl0):
    planes[k].model_fo(20)
    planes2[k].model_vr(20)
time.sleep(20)
n_pl1 = data.downlad()
planes_1 = [Plane(path, i) for i in range(0, n_pl1)]


a3 = plt.subplot()
colors = ['r','b','g','c','k','y','m','w','r','b','g','c','k','y','m','w','r','b','g','c','k','y','m','w','r','b','g','c','k','y','m','w']
for i in range(0,n_pl0):
    a3.scatter(planes_0[i].azi, planes_0[i].ele, color=colors[i], marker='o')
    a3.scatter(planes[i].azi, planes[i].ele, color=colors[i], marker='x', s=20)
    a3.scatter(planes2[i].azi, planes2[i].ele, color=colors[i], marker='*', s=20)
    for h in range(0,n_pl1):
        if planes[i].hex == planes_1[h].hex:
            a3.scatter(planes_1[h].azi, planes_1[h].ele, color=colors[i], marker='+',s=15)
            print(planes_0[i].speed,planes[i].speed, planes_1[h].speed)
a3.set_xlim(0, 360)
a3.set_ylim(0,90)
a3.set_title('Modeling Error')
a3.grid()
a3.set_xlabel("Azimut [deg]")
a3.set_ylabel("Elevation [deg]")
plt.show()

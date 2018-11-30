import requests, json, datetime, time, csv
import numpy as np
import io, copy
import sys, logging
from matplotlib import pyplot as plt
from scipy import optimize
import matplotlib.cm as cm


t0 = time.time()
dkm = 40000.0 / 360.0
dt = 10
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

    def grid2pos(self, g):  # g = [a,e]  grid indices
        azi = 47 + 4.5*g[0]
        ele = 15 + 4.5*g[1]
        return [azi, ele]

    def pos2grid(self, pos):
        a = np.arange(45 + 2, 314, 4.5)
        e = np.arange(13 + 2, 81, 4.5)
        e_idx = np.argmin(np.abs(pos[1]-e))
        a_idx = np.argmin(np.abs(pos[0]-a))
        return np.array([a_idx, e_idx])

    def observe(self):
        self.sky[self.pos2grid([self.azi, self.ele])[0], self.pos2grid([self.azi, self.ele])[1]] += dt

    def mult_obs(self, x_f):
        g0 = self.pos2grid([T.azi, T.ele])
        p_a = np.linspace(T.azi, x_f[0], 100)
        p_e = np.linspace(T.ele, x_f[1], 100)
        i = 0
        r = 0
        while i < 100:
            if all(g0 == self.pos2grid([p_a[i], p_e[i]])):
                i += 1
            else:
                self.sky[g0[0], g0[1]] += dt*(i-r)/100
                g0 = self.pos2grid([p_a[i], p_e[i]])
                r = i
                i += 1
        self.sky[g0[0], g0[1]] += dt * (i-r) / 100

    def cost_fct(self, x):
        p1 = 0
        i = self.pos2grid(x[0:2])
        im = self.pos2grid(x[2:4])
        sky_temp = copy.copy(self.sky)
        sky_temp[i[0], i[1]] += dt
        for pl in planes:
            d = np.linalg.norm(np.array([x[0] - pl.azi, x[1] - pl.ele]))
            if d < 10:
                p1 += 20*1/d                                               # TODO: tune parameters
        if self.sky[i[0], i[1]] > 30 + (time.time()-t0)/100:
            p1 += 0.1*self.sky[i[0], i[1]]
        p1 += np.linalg.norm(np.array(x[0]-self.azi, x[1]-self.ele))
        p2 = 0
        for plm in planes_mod:
            d = np.linalg.norm(np.array([x[2] - plm.azi, x[3] - plm.ele]))
            if d < 10:
                p2 += 10*1/d                                               # TODO: tune parameters
        if self.sky[i[0], i[1]] > 30 + (time.time()-t0)/100:
            p2 += 0.1*sky_temp[im[0], im[1]]
        p2 += np.linalg.norm(np.array(x[2]-x[0], x[3]-x[1]))
        return p1 + 0.5*p2

    def mpc(self):
        o = []
        v = []
        dx = self.max_speed * dt - 0.1
        dx0 = [[0, 0, 0, 0], [dx, 0, 0, 0], [-dx, 0, 0, 0], [0, dx, 0, 0], [0, -dx, 0, 0]]
        for d in dx0:
            x0 = np.array([self.azi, self.ele, self.azi, self.ele]) + d
            bnds = ((45, 314), (13, 81), (45, 314), (13, 81))
            cons = ({'type': 'ineq',
                     'fun': lambda x: self.max_speed * dt - np.linalg.norm(np.array([T.azi - x[0], T.ele - x[1]]))},
                    {'type': 'ineq',
                     'fun': lambda x: self.max_speed * dt - np.linalg.norm(np.array([x[0] - x[2], x[1] - x[3]]))})
            v.append(optimize.minimize(self.cost_fct, x0, constraints=cons, bounds=bnds).fun)
            o.append(optimize.minimize(self.cost_fct, x0, constraints=cons, bounds=bnds).x)
        b = np.nanargmin(v)
        return o[b]

    def move(self, x):
        logging.basicConfig(filename="ctrl_output.log", level=logging.INFO, format="%(asctime)-15s  %(message)s")
        a = str('%.3f' % x_opt[0])
        e = str('%.3f' % x_opt[1])
        logging.info("azi " + a)
        logging.info("ele " + e)
        # TODO: write output with new azi, ele to move the real telescope

    def show_obs(self):
        o = plt.gca()
        #o = fig.add_subplot(212)
        plt.gca()
        a = np.size(T.sky, 0)
        e = np.size(T.sky, 1)
        o.set_xlim(0, 360)
        o.set_ylim(0, 90)
        o.set_xlabel('Azimut')
        o.set_ylabel('Elevation')
        o.set_title('Observations')
        r = 3.25
        for ia in range(0, a):
            for ie in range(0, e):
                if self.sky[ia, ie] != 0:
                    c = cm.YlOrBr(self.sky[ia, ie] / (dt * 10))
                    pos = self.grid2pos([ia, ie])
                    circle = plt.Circle(xy=(pos[0], pos[1]), radius=r, color=c)
                    o.add_artist(circle)
        return o


class Plane:

    def __init__(self, path, i):

        ff = open(path, 'r')
        strings = np.genfromtxt(ff, delimiter=",", dtype=str)
        ff.close()
        ff = open(path, 'r')
        numbers = np.genfromtxt(ff, delimiter=",", dtype=float, usecols=np.arange(3, 11))
        ff.close()
        self.hex = strings[i][1]
        self.lat = numbers[i, 0]
        self.long = numbers[i, 1]
        self.alt = numbers[i, 2]/3.2808/1000.0
        self.speed = numbers[i, 5]*1.8520/3600.0  # [knots] --> [km/s]  may be mph => *1.609344
        self.dir = numbers[i, 4]*np.pi/180
        self.vert_rate = numbers[i, 3]*0.3048/60.0/1000.0  # vert_rate [ft/min]->[km/s]
        self.rssi = numbers[i, 6]
        self.pos = np.array([numbers[i, 0]*dkm, numbers[i, 1]*dkm, numbers[i, 2]/3.2808/1000.0])
        self.R = np.linalg.norm(self.pos-np.array([47.33982*dkm, 8.1112*dkm, 0.540]), axis=0)  # absolute distance
        self.prev_speed = self.speed
        self.pr_vert_rate = self.vert_rate
        self.ele = np.arcsin((self.alt-0.54)/self.R)/np.pi*180.0
        self.azi = 180.0 + np.arctan2((self.pos[0]-47.33982*dkm), (self.pos[1])-8.111203*dkm)/np.pi*180.0
        self.last_seen = numbers[i, 7]
        self.mod = 0
        self.model_fo(self.last_seen)

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
        self.azi = 180.0 + np.arctan2((self.pos[0]-47.33982*dkm), (self.pos[1])-8.111203*dkm)/np.pi*180.0
        self.last_seen = 0
        self.mod = self.mod + 1


def plot_error():
    n = len(planes_temp)
    nf = len(planes)
    a3 = fig.add_subplot(111)
    colors = ['r', 'b', 'g', 'c', 'k', 'y', 'm', 'w']
    for i in range(1, 5):
        colors += colors
    a3.scatter(T.azi, T.ele, marker='^', s=60)
    #a3.plot(T.azi + x, T.ele + y, 'b-')
    for i in range(0, n):
        a3.scatter(planes_temp[i].azi, planes_temp[i].ele, color=colors[i], marker='+')
        a3.scatter(planes_mod[i].azi, planes_mod[i].ele, color=colors[i], marker='x', s=20)
        for h in range(0, nf):
            if planes_temp[i].hex == planes[h].hex:
                a3.scatter(planes[h].azi, planes[h].ele, color=colors[i], marker='o', s=15)
    a3.set_xlim(0, 360)
    a3.set_ylim(0, 90)
    a3.set_title('Real time model')
    a3.set_xlabel("Azimut [deg]")
    a3.set_ylabel("Elevation [deg]")
    a3.legend(['Beam radius', 'Telescope', 'Previous position', 'Modeled position', 'Actual position'])
    a3.grid()
    return a3


T = Telescope(120, 13.1)  # position initialized
data = Data(url, path)
first = 1
fig = plt.figure(figsize=(20, 5))
theta = np.arange(0, 2 * np.pi + np.pi / 50, np.pi / 50)
x = 3.250 * np.array([np.cos(q) for q in theta])
y = 3.250 * np.array([np.sin(q) for q in theta])
t0 = time.time()
while True:
    print(time.time()-t0)
    n_pl = data.downlad()                                    # planes=new real  planes_mod=new modeled  planes_temp=old
    planes = [Plane(path, i) for i in range(0, n_pl)]
    if first != 1:
        fig.clear()
        plt.ion()
        o = T.show_obs()
        a = plot_error()
        cbar_ax = fig.add_axes([0.93, 0.1, 0.01, 0.8])
        sc = plt.scatter(0, 0, s=0, c=0.5, cmap='YlOrBr', vmin=0, vmax=dt*10, facecolors='none')
        cbar = plt.colorbar(sc, cax=cbar_ax)
        cbar.set_label('S',rotation=0, labelpad=10)
        plt.pause(0.1)
        plt.grid()
        fig.canvas.draw()
        for k in range(0, n_t):
            if planes_temp[k].hex not in [planes[j].hex for j in range(0, n_pl)] and planes_temp[k].R < max_dist \
                    and planes_temp[k].mod < 2:
                np.append(planes, planes_temp[k].model_fo(dt))
            else:
                for j in range(0, n_pl):
                    if planes_temp[k].hex == planes[j].hex:
                        planes[j].prev_speed = planes_temp[k].speed
                        planes[j].pr_vert_rate = planes_temp[k].vert_rate
    first += 1
    planes_temp = []
    planes_mod = []
    for j in range(0, n_pl):
        planes_temp = np.append(planes_temp, copy.copy(planes[j]))
        pm = copy.copy(planes[j])
        pm.model_fo(dt)
        planes_mod = np.append(planes_mod, pm)

    n_t = n_pl
    x_opt = T.mpc()
    dist = []
    mov = np.linalg.norm(np.array([T.azi - x_opt[0], T.ele - x_opt[1]]))
    for pl in planes:
        dist.append(np.linalg.norm(np.array([T.azi - pl.azi, T.ele - pl.ele])))
    m = min(dist)
    ind = np.nanargmin(dist)
    if mov < 0.1:
        print('\nNo movement needed, closest plane at:', str('%.2f' % m),
              'degrees,  in: [', str('%.3f' % planes[ind].azi), ',', str('%.3f' % planes[ind].ele), ']')
    else:
        print('\nMoving from:', [T.azi, T.ele], 'to:', x_opt[0:2])
        if m < 10:
            print('Caused by planes   in: [', str('%.3f' % planes[ind].azi), ',', str('%.3f' % planes[ind].ele), ']')
        else:
            print('Caused by observation priority')
    grid1 = T.pos2grid([T.azi, T.ele])
    grid2 = T.pos2grid(x_opt[0:2])
    if T.azi != x_opt[0] and T.ele != x_opt[1]:
        if all(grid1 == grid2):
            T.observe()
        else:
            T.mult_obs(x_opt)

        T.azi = x_opt[0]
        T.ele = x_opt[1]
        T.move(x_opt)
    else:
        T.observe()
    #T.show_obs()

    time.sleep(dt)


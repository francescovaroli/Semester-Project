import requests, json, datetime, time, csv
import ftplib
import numpy as np
import copy
import logging
from matplotlib import pyplot as plt
from scipy import optimize
import matplotlib.cm as cm
import ephem


t0 = time.time()
dkm = 40000.0 / 360.0
dt = 60
url = 'http://129.132.63.205:8080/aircraft.json'
path_mat = "/home/francesco/PycharmProjects/SemesterProject/data/mapMatrix.csv"
path = '/home/francesco/PycharmProjects/SemesterProject/data/dataOUT.csv'
ff = open(path_mat, 'r')
map = np.genfromtxt(ff, delimiter=",")

z = [47.451542, 8.564572, 432.0/1000.0]
b = [47.444, 8.2296, 396.24/1000.0]
max_dist = 500

def notNan(n):
    return n == n


def degDist(p1, p2):
    return np.linalg.norm([p1[0] - p2[0], p1[1] - p2[1]])



class Data:
    def __init__(self, urld, file_path):
        self.url = urld
        self.file_path = file_path

    def download(self):
        n = 0
        try:
            new_data = json.loads(requests.get(self.url).content)['aircraft']
            now = str(datetime.datetime.now())
            fields = ['time', 'hex', 'flight', 'lat', 'lon', 'altitude', 'vert_rate',
                      'track', 'speed', 'rssi', 'seen_pos']
            with open(self.file_path, 'w') as writeFile:
                writer = csv.DictWriter(writeFile, fieldnames=fields)
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
                        if 'speed' not in line:
                            line['speed'] = 0
                        if 'vert_rate' not in line:
                            line['vert_rate'] = 0
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
        self.sky = np.zeros((80, 20))
        self.sky_cel = np.zeros((80, 40))

    def coordview(self, pos):
        R = np.linalg.norm(np.array([pos[0]*dkm, pos[1]*dkm, pos[2]])-np.array([47.33982*dkm, 8.1112*dkm, 0.540]), axis=0)
        ele = np.arcsin((pos[2] - 0.54) / R) / np.pi * 180.0
        at = np.arctan2((pos[1] - 8.111203), (pos[0] - 47.33982))/ np.pi * 180.0
        azi = -180.0*(np.sign(at)-1) + at
        return azi, ele

    def grid2pos(self, g):  # g = [a,e]  grid indices
        azi = 4.5*g[0]
        ele = 4.5*g[1]
        return [azi, ele]

    def pos2grid(self, pos):
        a = np.arange(0, 360, 4.5)
        e = np.arange(0, 90, 4.5)
        e_idx = np.argmin(np.abs(pos[1]-e))
        a_idx = np.argmin(np.abs(pos[0]-a))
        return np.array([a_idx, e_idx])

    def grid2pos_c(self, g):  # g = [a,e]  grid indices
        azi = -180 + 4.5*g[0]
        ele = -90 + 4.5*g[1]
        return [azi, ele]

    def pos2grid_c(self, pos):
        a = np.arange(-180, 180, 4.5)
        e = np.arange(-90, 90, 4.5)
        e_idx = np.argmin(np.abs(pos[1]-e))
        a_idx = np.argmin(np.abs(pos[0]-a))
        return np.array([a_idx, e_idx])

    def terr2cel(self, a, e, t):
        observer = ephem.Observer()
        observer.lon = self.long
        observer.lat = self.lat
        observer.elevation = self.alt
        observer.date = ephem.now() + t/(3600*24)
        ra, d = observer.radec_of(a/180*np.pi, e/180*np.pi)
        return ra*180/np.pi, d*180/np.pi

    def observe(self):
        self.sky[self.pos2grid([self.azi, self.ele])[0], self.pos2grid([self.azi, self.ele])[1]] += dt
        ra, d = self.terr2cel(self.azi, self.ele, 0)
        self.sky_cel[self.pos2grid_c([ra, d])[0], self.pos2grid_c([ra, d])[1]] += dt

    def mult_obs(self, x_f):
        x_cel = self.terr2cel(x_f[2], x_f[3], 0)
        ra, d = self.terr2cel(self.azi, self.ele, 0)
        g0 = self.pos2grid([T.azi, T.ele])
        g0_cel = self.pos2grid_c([ra, d])
        p_a = np.linspace(T.azi, x_f[2], 100)
        p_e = np.linspace(T.ele, x_f[3], 100)
        p_a_cel = np.linspace(ra, x_cel[0], 100)
        p_e_cel = np.linspace(d, x_cel[1], 100)
        i = 0
        j = 0
        r = 0
        q = 0
        while i < 100:
            if all(g0 == self.pos2grid([p_a[i], p_e[i]])):
                i += 1
            else:
                self.sky[g0[0], g0[1]] += dt*(i-r)/100
                g0 = self.pos2grid([p_a[i], p_e[i]])
                r = i
                i += 1
        self.sky[g0[0], g0[1]] += dt * (i-r) / 100

        while j < 100:
            if all(g0_cel == self.pos2grid_c([p_a_cel[j], p_e_cel[j]])):
                j += 1
            else:
                self.sky_cel[g0_cel[0], g0_cel[1]] += dt * (j - q) / 100
                g0_cel = self.pos2grid_c([p_a_cel[j], p_e_cel[j]])
                q = j
                j += 1
        self.sky_cel[g0_cel[0], g0_cel[1]] += dt * (j-q) / 100

    def cost_fct(self, x):
        p1 = 0
        i = self.pos2grid_c(self.terr2cel(x[0], x[1], 1.5*dt))
        im = self.pos2grid_c(self.terr2cel(x[2], x[3], 2*dt))
        j = self.pos2grid(x[0:2])
        jm = self.pos2grid(x[2:4])
        p1 += map[jm[0], j[1]]/20
        for pl in planes_mod1:
            d = degDist([x[0], x[1]], [pl.azi, pl.ele])
            if d < 15:
                p1 += 50*1/d                                               # TODO: tune parameters
        if self.sky_cel[i[0], i[1]] > 200 + (time.time()-t0)/100:
            p1 += 0.01*self.sky_cel[i[0], i[1]]
        p1 += 0.3*degDist([x[0], x[1]], [self.azi, self.ele])
        p2 = 0
        p2 += map[jm[0], jm[1]]/100
        for plm in planes_mod2:
            d = degDist([x[2], x[3]], [plm.azi, plm.ele])
            if d < 15:
                p2 += 50*1/d                                               # TODO: tune parameters
        if self.sky_cel[i[0], i[1]] > 200 + (time.time()-t0)/100:
            p2 += 0.01*self.sky_cel[im[0], im[1]]
        p2 += 0.3*degDist([x[2], x[3]], [x[0], x[1]])
        return 0.5*p1 + p2

    def mpc(self):
        o = []
        v = []
        dx = self.max_speed * 0.5*dt - 0.5
        dx0 = [[0, 0, 0, 0], [dx, 0, dx, 0], [-dx, 0, -dx, 0], [0, dx, 0, dx], [0, -dx, 0, -dx]]
        for d in dx0:
            x0 = np.array([self.azi, self.ele, self.azi, self.ele]) + d
            bnds = ((45, 314), (20, 51), (45, 314), (20, 51))
            cons = ({'type': 'ineq',
                     'fun': lambda x: self.max_speed * 0.5*dt - degDist([x[0], x[1]], [T.azi, T.ele])},
                    {'type': 'ineq',
                     'fun': lambda x: self.max_speed * 0.5*dt - degDist([x[2], x[3]], [x[0], x[1]])})
            v.append(optimize.minimize(self.cost_fct, x0, constraints=cons, bounds=bnds).fun)
            o.append(optimize.minimize(self.cost_fct, x0, constraints=cons, bounds=bnds).x)
        b = np.nanargmin(v)
        return o[b]

    def store_interference(self):
        logging.basicConfig(filename="Interferences.log", level=logging.INFO, format="%(asctime)-15s  %(message)s")
        here = str([T.azi, T.ele])
        logging.info("Corrupted observation in" + here)

    def move(self, x, d):

        a = str(x[0])
        e = str(x[1])
        s = open("/home/francesco/PycharmProjects/SemesterProject/data/scheduler.txt", 'w')
        s.write(str(d.year)+'-'+ d.strftime("%m")+'-'+ d.strftime("%d")+', '+ d.strftime("%H")+':'+ d.strftime("%M")+', '+a+', '+e+', Control')
        s.close()
        file = open("/home/francesco/PycharmProjects/SemesterProject/data/scheduler.txt", 'rb')
        session = ftplib.FTP('pavo.ethz.ch', 'ADS', 'ads-B007')
        session.storbinary('STOR scheduler_prv.txt', file)
        file.close()
        session.quit()
        # TODO: +1 minute

    def show_obs(self, x):
        o = fig.add_subplot(211)
        o = plt.gca()
        plt.gca()
        a = np.size(T.sky, 0)
        e = np.size(T.sky, 1)
        o.set_xlim(0, 360)
        o.set_ylim(0, 90)
        o.set_xlabel('Azimuth')
        o.set_ylabel('Elevation')
        o.set_title('Observations ' + str(time.ctime()))
        o.grid()
        r = 3.25
        za, ze = self.coordview(z)
        ba, be = self.coordview(b)
        n = len(planes_temp)
        nf = len(planes)
        colors = ['r', 'b', 'g', 'c', 'k', 'y', 'm', 'w']
        for i in range(1, 5):
            colors += colors
        o.scatter(za, ze, marker='^', color='g', s=60)
        o.scatter(ba, be, marker='^', color='b', s=60)
        o.scatter(x[2], x[3], marker='+', color='y', s=60)
        for i in range(0, n):
            o.scatter(planes_mod1[i].azi, planes_mod1[i].ele, color=colors[i], marker='s', s=20)
            o.scatter(planes_mod2[i].azi, planes_mod2[i].ele, color=colors[i], marker='p', s=20)
            for h in range(0, nf):
                if planes_temp[i].hex == planes[h].hex:
                    o.scatter(planes[h].azi, planes[h].ele, color=colors[i], marker='o', s=15)

        for ia in range(0, a):
            for ie in range(0, e):
                if self.sky[ia, ie] != 0:
                    c = cm.YlOrBr(self.sky[ia, ie] / (dt * 10))
                    pos = self.grid2pos([ia, ie])
                    circle = plt.Circle(xy=(pos[0], pos[1]), radius=r, color=c)
                    o.add_artist(circle)
        o.legend(['Zurich airport', 'Birrfeld airport','Optimum', '1.5dt modeled position','2dt modeled position', 'Previous position'])
        return o


    def show_cel(self):
        s = fig.add_subplot(212, projection='aitoff')
        a = np.size(T.sky_cel, 0)
        e = np.size(T.sky_cel, 1)
        #s.set_xlim(-180, 180)
        #s.set_ylim(-90, 90)
        r = 3.25
        for ia in range(0, a):
            for ie in range(0, e):
                if self.sky_cel[ia, ie] != 0:
                    c = cm.YlOrBr(self.sky_cel[ia, ie] / (dt * 10))
                    pos = self.grid2pos_c([ia, ie])
                    s.scatter(pos[0]/180*np.pi, pos[1]/180*np.pi, c=c, marker='o', s=50)
                    s.scatter(self.terr2cel(T.azi, T.ele, 0)[0]/180*np.pi, self.terr2cel(T.azi, T.ele, 0)[1]/180*np.pi, c='g', marker='o', s=10)

        s.legend(['Observations intensity',' Present observation'])
        s.set_title('\nObservations in celestial coordinates\n')
        s.set_xlabel('Right ascension')
        s.set_ylabel('Declination')
        s.grid()
        return s

    def plots(self, x):
        fig.clear()
        fig.subplots_adjust(hspace=0.3)
        plt.ion()
        o = T.show_obs(x)
        s = T.show_cel()
        cbar_ax = fig.add_axes([0.93, 0.1, 0.01, 0.8])
        sc = plt.scatter(0, 0, s=0, c=0.5, cmap='YlOrBr', vmin=0, vmax=dt * 10, facecolors='none')
        cbar = plt.colorbar(sc, cax=cbar_ax)
        cbar.set_label('S', rotation=0, labelpad=10)
        plt.pause(0.1)
        plt.grid()
        fig.canvas.draw()


class Planes:

    def __init__(self, path):

        ff = open(path, 'r')
        self.strings = np.genfromtxt(ff, delimiter=",", dtype=str)
        ff.close()
        ff = open(path, 'r')
        self.numbers = np.genfromtxt(ff, delimiter=",", dtype=float, usecols=np.arange(3, 11))
        ff.close()


class Plane:
    def __init__(self, i, strings, numbers):
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
        at = np.arctan2((self.pos[1] - 8.111203*dkm), (self.pos[0] - 47.33982*dkm))/ np.pi * 180.0
        self.azi = -180.0*(np.sign(at)-1) + at
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
        at = np.arctan2((self.pos[1] - 8.111203*dkm), (self.pos[0] - 47.33982*dkm))/ np.pi * 180.0
        self.azi = -180.0*(np.sign(at)-1) + at
        self.last_seen = 0
        self.mod = self.mod + 1

    def modelAzEl(self,t):
        dv = self.speed - self.prev_speed
        dvr = self.vert_rate - self.pr_vert_rate
        mlat = self.lat + (self.speed + 0.5 * dv) * np.cos(self.dir) * t / dkm
        mlong = self.long + (self.speed + 0.5 * dv) * t * np.sin(self.dir) / dkm
        malt = self.alt + t * (self.vert_rate + 0.5 * dvr)
        mpos = np.array([mlat * dkm, mlong * dkm, malt])
        mR = np.linalg.norm(mpos - np.array([47.33982 * dkm, 8.1112 * dkm, 0.540]), axis=0)  # absolute distance
        mele = np.arcsin((malt - 0.54) / mR) / np.pi * 180.0
        mat = np.arctan2((mlong - 8.111203), (mlat - 47.33982 )) / np.pi * 180.0
        mazi = -180.0 * (np.sign(mat) - 1) + mat
        return [mazi, mele]


T = Telescope(np.float64(50), np.float64(20))  # position initialized
data = Data(url, path)
fig = plt.figure(figsize=(20, 6))
first = 1
t0 = time.time()

while True:
    ti = time.time()
    print('\n', time.ctime())
    n_pl = data.download()  # planes=new real  planes_mod1=new modeled  planes_temp=old
    P = Planes(path)
    planes = [Plane(i, P.strings, P.numbers) for i in range(0, n_pl)]
    if first != 1:
        grid1 = T.pos2grid([T.azi, T.ele])
        grid2 = T.pos2grid(x_opt[2:4])
        if T.azi != x_opt[2] or T.ele != x_opt[3]:
            if all(grid1 == grid2):
                T.observe()
            else:
                T.mult_obs(x_opt)

            T.azi = x_opt[2]
            T.ele = x_opt[3]
        else:
            T.observe()

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
    planes_mod1 = []
    planes_mod2 = []
    for j in range(0, n_pl):
        planes_temp = np.append(planes_temp, copy.copy(planes[j]))
        pm = copy.copy(planes[j])
        pm.model_fo(1.5*dt)
        planes_mod1 = np.append(planes_mod1, pm)
        pm2 = copy.copy(planes[j])
        pm2.model_fo(2*dt)
        planes_mod2 = np.append(planes_mod2, pm2)
    n_t = n_pl
    x_opt = T.mpc()

    T.plots(x_opt)

    dist = []
    for pl in planes_mod2:
        dist.append(degDist([pl.azi,  pl.ele], [x_opt[2], x_opt[3]]))
    m = min(dist)
    ind = np.nanargmin(dist)
    mov = degDist([T.azi, T.ele], [x_opt[2], x_opt[3]])
    if mov < 0.1:
        print('No movement from ', T.azi.round(2), T.ele.round(2), ' needed\nClosest plane at:', str('%.2f' % m),
              'degrees,  in: [', str('%.3f' % planes_mod2[ind].azi), ',', str('%.3f' % planes_mod2[ind].ele), ']')
    else:
        print('Moving from:', [T.azi.round(2), T.ele.round(2)], 'to:', x_opt[2:4].round(2), 'dist = ', mov)
        if m < 15:
            print('Caused by planes   in: [', str('%.3f' % planes_mod2[ind].azi), ',', str('%.3f' % planes_mod2[ind].ele), ']')
            if m < 3.5:
                T.store_interference()
        else:
            print('Caused by observation priority: closest plane at:', str('%.2f' % m),' degrees')
    to = time.time()
    if T.azi != x_opt[1] or T.ele != x_opt[2]:
        T.move(x_opt, datetime.datetime.today())
    time.sleep(dt-(to-ti))


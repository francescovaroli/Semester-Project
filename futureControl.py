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
url_data = 'http://129.132.63.205:8080/aircraft.json'
path_prior_observation = "/home/francesco/PycharmProjects/SemesterProject/data/mapMatrix.csv"
path_planes_data = '/home/francesco/PycharmProjects/SemesterProject/data/dataOUT.csv'
file_prior_observation = open(path_prior_observation, 'r')
map_prior_observation = np.genfromtxt(file_prior_observation, delimiter=",")

zurich_coordinates = [47.451542, 8.564572, 432.0/1000.0]
birrfeld_coordinates = [47.444, 8.2296, 396.24/1000.0]
max_dist = 500

def not_Nan(n):
    ''' True if not a nan'''
    return n == n


def deg_dist(p1, p2):
    ''' Distance between two objects'''
    return np.linalg.norm([p1[0] - p2[0], p1[1] - p2[1]])


class Data:
    def __init__(self, urld, file_path):
        self.url = urld
        self.file_path = file_path

    def download(self):
        '''
        Downloads airplnes data and store them in a csv file at file_path on local omputer
        :return: number of planes
        '''
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
    '''
    To keep track of the observed areas and plot current state
    '''
    def __init__(self, azi, ele):
        self.lat = 47.33982
        self.long = 8.111203
        self.y = self.long*dkm
        self.x = self.lat*dkm
        self.alt = 540.0/1000
        self.azi = azi
        self.ele = ele
        self.max_azi = 4.6
        self.max_ele = 2.2
        self.beam = 3.25*np.pi/180  # degrees of the beam cone
        self.dir = [[self.lat, (self.lat + np.cos(self.azi*np.pi/180))],
                    [self.long, (self.long + np.sin(self.azi*np.pi/180))],
                    [self.alt, self.alt + (np.sin(self.ele*np.pi/180))*dkm]]
        self.sky = np.zeros((80, 20))
        self.sky_cel = np.zeros((80, 40))

    def coordview(self, pos):
        '''
        Transform from world to telescope frame
        :param pos: plane position in world frame[lat, long]
        :return: [azi, ele] in telescope frame
        '''
        R = np.linalg.norm(np.array([pos[0]*dkm, pos[1]*dkm, pos[2]])-np.array([47.33982*dkm, 8.1112*dkm, 0.540]), axis=0)
        ele = np.arcsin((pos[2] - 0.54) / R) / np.pi * 180.0
        at = np.arctan2((pos[1] - 8.111203), (pos[0] - 47.33982))/ np.pi * 180.0
        azi = -180.0*(np.sign(at)-1) + at
        return azi, ele

    def grid2pos(self, g):  # g = [a,e]  grid indices
        '''
        transform from grid indeces to coordinates
        :param g: grid indeces
        :return: [azi, ele] coordinates
        '''
        azi = 4.5*g[0]
        ele = 4.5*g[1]
        return [azi, ele]

    def pos2grid(self, pos):
        '''
        transform from coordinates to grid indeces
        :param pos: coordinates
        :return: [i, j] grid indeces
        '''
        a = np.arange(0, 360, 4.5)
        e = np.arange(0, 90, 4.5)
        e_idx = np.argmin(np.abs(pos[1]-e))
        a_idx = np.argmin(np.abs(pos[0]-a))
        return np.array([a_idx, e_idx])

    def grid2pos_c(self, g):  # g = [a,e]  grid indices
        '''
        transform from grid indeces to celestial coordinates
        :param g: grid indeces
        :return: [azi, ele] celestial coordinates
        '''
        azi = -180 + 4.5*g[0]
        ele = -90 + 4.5*g[1]
        return [azi, ele]

    def pos2grid_c(self, pos):
        '''
        transform from celestial coordinates to grid indeces
        :param pos: celestial coordinates
        :return: [i, j] grid indeces
        '''
        a = np.arange(-180, 180, 4.5)
        e = np.arange(-90, 90, 4.5)
        e_idx = np.argmin(np.abs(pos[1]-e))
        a_idx = np.argmin(np.abs(pos[0]-a))
        return np.array([a_idx, e_idx])

    def terr2cel(self, a, e, t):
        '''
        Transform from terrestrial to celestial coordinates
        :param a: azimuth
        :param e: elevation
        :param t: time
        :return: right ascension, declination
        '''
        observer = ephem.Observer()
        observer.lon = self.long
        observer.lat = self.lat
        observer.elevation = self.alt
        observer.date = ephem.now() + t/(3600*24)
        ra, dec = observer.radec_of(a/180*np.pi, e/180*np.pi)
        return ra*180/np.pi, dec*180/np.pi

    def sun_position(self):
        '''
        :return: [azimut, elevation] of the sun
        '''
        observer = ephem.Observer()
        observer.lon = self.long
        observer.lat = self.lat
        observer.date = ephem.now()
        sun = ephem.Sun(observer)
        sun.compute(observer)
        return [sun.az*180/np.pi,sun.alt*180/np.pi]


    def observe(self):
        '''
        observe current spot of the sky
        :return:
        '''
        self.sky[self.pos2grid([self.azi, self.ele])[0], self.pos2grid([self.azi, self.ele])[1]] += dt
        ra, dec = self.terr2cel(self.azi, self.ele, 0)
        self.sky_cel[self.pos2grid_c([ra, dec])[0], self.pos2grid_c([ra, dec])[1]] += dt

    def mult_obs(self, x_f):
        '''
        observe multiple spots of the sky while mooving
        :param x_f: final position
        :return:
        '''
        x_cel = self.terr2cel(x_f[2], x_f[3], 0)
        ra, dec = self.terr2cel(self.azi, self.ele, 0)
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
        '''
        cost of a sky spot weighted on present and future states
        :param x:
        :return:
        '''
        q = [0.5, 1, 0.5,0.5]
        p = 0
        for k in [0,1,2,3]:
            p += q[k]*self.cost(x[2*k:(2*k+2)], k)

        return p

    def cost(self, x, k):
        '''
        cost function per istant
        :param x: spot of the sky
        :param k: time-steps distance from present
        :return: cost of the spot
        '''
        cost = 0
        i = self.pos2grid_c(self.terr2cel(x[0], x[1],  (1.5+k/2) * dt))
        j = self.pos2grid(x)
        cost += map_prior_observation[j[0], j[1]] / 20
        cost += max(0, x[1]-40)/2
        for plane in planes_mod[k]:

            distance = deg_dist([x[0], x[1]], [plane.azi, plane.ele])
            if distance < 35:
                cost += 25/np.abs(plane.rssi) * 50 * 1 / (l+1)
            plane_model = plane.modelAzEl(-dt/4)
            d1 = deg_dist([x[0], x[1]], [plane_model[0], plane_model[1]])
            if d1 < 35:
                cost += 25/np.abs(pl.rssi) * 50 * 1 / (d1+1)
            if k == 0:
                previous_plane_model = plane.modelAzEl(-dt / 2)
                d0 = deg_dist([x[0], x[1]], [previous_plane_model[0], previous_plane_model[1]])
                if d0 < 35:
                    cost += 25 / np.abs(plane_model.rssi) * 50 * 1 / (d0 + 1)
        #if self.sky_cel[i[0], i[1]] > 200 + (time.time() - t0) / 100:
            #p1 += 0.01 * self.sky_cel[i[0], i[1]]
        cost += 0.3 * deg_dist([x[0], x[1]], [self.azi, self.ele])
        return cost

    '''def mpc(self):
        o = []
        v = []
        dx0 = [[0, 0, 0, 0, 0, 0], [self.max_azi, 0, self.max_azi, 0, self.max_azi, 0], [-self.max_azi, 0, -self.max_azi, 0, -self.max_azi, 0],
               [0, self.max_ele, 0, self.max_ele, 0, self.max_ele], [0, -self.max_ele, 0, -self.max_ele, 0, -self.max_ele]]
        ba = (45, 313)
        be = (20, 51)
        for d0 in dx0:
            x0 = np.array([self.azi, self.ele, self.azi, self.ele, self.azi, self.ele]) + d0
            bnds = (ba, be, ba, be, ba, be)
            cons = ({'type': 'ineq', 'fun': lambda x: self.max_azi/2 - abs(x[0] - T.azi)},
                    {'type': 'ineq', 'fun': lambda x: self.max_ele/2 - abs(x[1] - T.ele)},
                    {'type': 'ineq', 'fun': lambda x: self.max_azi/2 - abs(x[2] - x[0])},
                    {'type': 'ineq', 'fun': lambda x: self.max_ele/2 - abs(x[3] - x[1])},
                    {'type': 'ineq', 'fun': lambda x: self.max_azi/2 - abs(x[4] - x[2])},
                    {'type': 'ineq', 'fun': lambda x: self.max_ele/2 - abs(x[5] - x[3])})

            v.append(optimize.minimize(self.cost_fct, x0, constraints=cons, bounds=bnds).fun)
            o.append(optimize.minimize(self.cost_fct, x0, constraints=cons, bounds=bnds).x)
        b = np.nanargmin(v)
        return o[b]

    def mpc1(self):
        x0 = np.array([self.azi, self.ele, self.azi, self.ele, self.azi, self.ele])
        ba = (45, 314)
        be = (20, 51)
        bnds = (ba, be, ba, be, ba, be)
        cons = ({'type': 'ineq', 'fun': lambda x: self.max_azi/2 - abs(x[0] - T.azi)},
                {'type': 'ineq', 'fun': lambda x: self.max_ele/2 - abs(x[1] - T.ele)},
                {'type': 'ineq', 'fun': lambda x: self.max_azi/2 - abs(x[2] - x[0])},
                {'type': 'ineq', 'fun': lambda x: self.max_ele/2 - abs(x[3] - x[1])},
                {'type': 'ineq', 'fun': lambda x: self.max_azi/2 - abs(x[4] - x[2])},
                {'type': 'ineq', 'fun': lambda x: self.max_ele/2 - abs(x[5] - x[3])})
        return optimize.minimize(self.cost_fct, x0, constraints=cons, bounds=bnds).x

'''
    def mpc2(self):
        '''
        optimization function
        to avoid local optima the optimization is initialized in five points inside the reachale range and the results are then compared
        :return: optimal spot
        '''
        v = []
        xs0 = []
        for a in [0, self.max_azi-0.1, -self.max_azi-0.1]:
            for e in [0, self.max_ele-0.1, -self.max_ele-0.1]:
                xs0.append([self.azi+a, self.ele+e])
                v.append(self.cost_fct(np.array([self.azi+a, self.ele+e, self.azi+a, self.ele+e, self.azi+a, self.ele+e,self.azi+a, self.ele+e])))

        x0 = xs0[np.argmin(v)]
        ba = (45, 313)
        be = (20, 51)
        bnds = (ba, be, ba, be, ba, be, ba, be)
        cons = ({'type': 'ineq', 'fun': lambda x: self.max_azi/2 - abs(x[0] - T.azi)},
                {'type': 'ineq', 'fun': lambda x: self.max_ele/2 - abs(x[1] - T.ele)},
                {'type': 'ineq', 'fun': lambda x: self.max_azi/2 - abs(x[2] - x[0])},
                {'type': 'ineq', 'fun': lambda x: self.max_ele/2 - abs(x[3] - x[1])},
                {'type': 'ineq', 'fun': lambda x: self.max_azi/2 - abs(x[4] - x[2])},
                {'type': 'ineq', 'fun': lambda x: self.max_ele/2 - abs(x[5] - x[3])},
                {'type': 'ineq', 'fun': lambda x: self.max_azi/2 - abs(x[6] - x[4])},
                {'type': 'ineq', 'fun': lambda x: self.max_ele/2 - abs(x[7] - x[5])})
        return optimize.minimize(self.cost_fct, [x0,x0,x0,x0], constraints=cons, bounds=bnds).x  # , options={'eps': 1}

    def store_interference(self):
        '''
        save log bout interference event
        :return:
        '''
        logging.basicConfig(filename="Interferences.log", level=logging.INFO, format="%(asctime)-15s  %(message)s")
        here = str([T.azi, T.ele])
        logging.info("Corrupted observation in" + here)

    def move(self, msg_old, msg):
        '''
        move the telescope sending the control command
        :param msg_old: old command
        :param msg: actual comand
        :return:
        '''
        s = open("/home/francesco/PycharmProjects/SemesterProject/data/scheduler.txt", 'w')
        s.write(msg_old+'\n'+msg+'\n ')
        s.close()
        st = open("/home/francesco/PycharmProjects/SemesterProject/data/control_storage2.txt", 'a')
        st.write(msg)
        st.close()
        file = open("/home/francesco/PycharmProjects/SemesterProject/data/scheduler.txt", 'rb')
        session = ftplib.FTP('pavo.ethz.ch', 'ADS', 'ads-B007')
        session.delete('scheduler.txt')
        session.storbinary('STOR scheduler.tmp', file)
        file.close()
        time.sleep(1)
        session.rename('scheduler.tmp', 'scheduler.txt')
        session.quit()

    def show_cost(self, x):
        '''
        show cost function around x
        :param x: sky spot
        :return:
        '''
        n = 100
        grid = np.zeros([n, n])
        elevations = np.linspace(self.ele-15,self.ele+15, n)
        azimuts = np.linspace(self.azi-15, self.azi+15, n)
        teg = np.argmin(np.abs(self.ele - elevations))
        tag = np.argmin(np.abs(self.azi - azimuts))
        xeg = np.argmin(np.abs(x[3] - elevations))
        xag = np.argmin(np.abs(x[2] - azimuts))

        for i in range(n):
            for j in range(n):
                if abs(azimuts[i]-self.azi) < self.max_azi and abs(elevations[j]-self.ele) < self.max_ele:
                    grid[j, i] = self.cost_fct([azimuts[i], elevations[j], azimuts[i], elevations[j], azimuts[i], elevations[j], azimuts[i], elevations[j]])
                else:
                    grid[j, i] = 42
        jj = fig.add_subplot(212)
        im = jj.imshow(grid, cmap='jet', origin='lower', interpolation='nearest')
        plt.scatter(tag, teg, marker='*', c='w', s=60)
        plt.scatter(xag, xeg, marker='+', c='k', s=60)
        plt.gca()
        sp = 10
        plt.xticks(np.arange(0, n, n / sp + 1, int), np.linspace(self.azi-15, self.azi+15, 11).round(0))
        plt.yticks(np.arange(0, n, n / sp, int), np.arange(self.ele-15, self.ele+15, 3, int))
        plt.ylabel('Elevation')
        plt.xlabel('Azimuth')
        plt.title('Cost Function')
        cbar = plt.colorbar(im)
        cbar.set_label('Cost', rotation=90, labelpad=10)
        return im

    def show_obs(self, x):
        '''
        plot observations-intensity of the sky
        :param x: optimal spot
        :return:
        '''
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
        zurich_azi, zurich_ele = self.coordview(zurich_coordinates)
        birrfeld_azi, birrfeld_ele = self.coordview(birrfeld_coordinates)
        nf = len(planes)
        colors = ['r', 'b', 'g', 'c', 'k', 'y', 'm', 'w']
        for i in range(1, 5):
            colors += colors
        o.scatter(zurich_azi, zurich_ele, marker='^', color='g', s=60)
        o.scatter(birrfeld_azi, birrfeld_ele, marker='^', color='b', s=60)
        o.scatter(x[2], x[3], marker='+', color='k', s=60)
        for i in range(0, nf):
            o.scatter(planes_mod[0][i].azi, planes_mod[0][i].ele, color=colors[i], marker='s', s=20)
            o.scatter(planes_mod[1][i].azi, planes_mod[1][i].ele, color=colors[i], marker='p', s=20)
            o.scatter(planes_mod[2][i].azi, planes_mod[2][i].ele, color=colors[i], marker='.', s=20)
            o.scatter(planes_mod[3][i].azi, planes_mod[3][i].ele, color=colors[i], marker='.', s=10)
            o.scatter(planes[i].azi, planes[i].ele, color=colors[i], marker='o', s=15)

        for ia in range(0, a):
            for ie in range(0, e):
                if self.sky[ia, ie] != 0:
                    c = cm.YlOrBr(self.sky[ia, ie] / (dt * 10))
                    pos = self.grid2pos([ia, ie])
                    circle = plt.Circle(xy=(pos[0], pos[1]), radius=r, color=c)
                    o.add_artist(circle)
        o.legend(['Zurich airport', 'Birrfeld airport','Optimum', '1.5dt modeled position','2dt modeled position','2.5dt modeled position','3.5', 'Previous position'])
        return o


    def show_cel(self):
        '''
        show observation in celestial coordinates
        :return:
        '''
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
        '''
        plot previous plots
        :param x:
        :return:
        '''
        fig.clear()
        fig.subplots_adjust(hspace=0.3)
        plt.ion()
        o = T.show_obs(x)
        im =T.show_cost(x)
        #s = T.show_cel()
        cbar_ax = fig.add_axes([0.93, 0.1, 0.01, 0.8])
        sc = plt.scatter(0, 0, s=0, c=0.5, cmap='YlOrBr', vmin=0, vmax=dt * 10, facecolors='none')
        cbar = plt.colorbar(sc, cax=cbar_ax)
        cbar.set_label('S', rotation=0, labelpad=10)
        plt.pause(0.1)
        plt.grid()
        fig.canvas.draw()


class Planes:

    def __init__(self, path):
        '''
        retrieve all planes from the file
        '''
        file = open(path, 'r')
        self.strings = np.genfromtxt(file, delimiter=",", dtype=str)
        file.close()
        file = open(path, 'r')
        self.numbers = np.genfromtxt(file, delimiter=",", dtype=float, usecols=np.arange(3, 11))
        file.close()


class Plane:
    def __init__(self, i, strings, numbers):
        '''
        create each plane attributes
        :param i: plane index
        :param strings: string attributes of Planes
        :param numbers: numerical attributes of Planes
        '''
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

    def model_fo(self, t):
        '''
        modelling future plane position with constant acceleration ---> effective whe we have already seen the plane
        '''
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
        '''
        model new azimut and elevation
        :param t: time gap
        :return: modeled azimut and elevation
        '''

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

# ---- INITIALIZATION ---- #

fig = plt.figure(figsize=(20, 6))
d = time.gmtime(time.time() + 100)
initial_azimut = str(300)
initial_elevation = str(45)
msg_old = '##'
msg = str(d.tm_year) + '-' + time.strftime("%m", d) + '-' + time.strftime("%d", d) + ', ' + \
      time.strftime("%H",d) + ':' + time.strftime('%M', d) + ', ' + sa + ', ' + se + ', predicted observation\n'
data = Data(url_data, path_planes_data)
first = 1
T = Telescope(np.float64(initial_azimut), np.float64(initial_elevation))  # position initialized
T.move(msg_old, msg)


# ---- WAIT TO SYNCRONIZE WITH TELESCOPE CONTROL ---- #

while datetime.datetime.today().second < 56:
    time.sleep(0.1)

t0 = time.time()
print('\n', time.ctime())

# ---- START CONTROL LOOP ---- #
while True:
    ti = time.time()
    planes_number = data.download()  # planes=new real  planes_mod[0]=new modeled  planes_temp=old
    if planes_number > 1:
        P = Planes(path_planes_data)
        planes = [Plane(i, P.strings, P.numbers) for i in range(0, planes_number)]
        if first != 1:
            grid1 = T.pos2grid([T.azi, T.ele])
            grid2 = T.pos2grid(x_opt[2:4])
            if deg_dist([T.azi, T.ele], x_opt[2:4]) > 0.1:
                T.observe()

                '''if all(grid1 == grid2):
                    T.observe()
                else:
                    T.mult_obs(x_opt)'''

                T.azi = x_opt[2]
                T.ele = x_opt[3]
            else:
                T.observe()

            for k in range(0, n_t):
                if planes_temp[kS].hex not in [planes[j].hex for j in range(0, planes_number)] and planes_temp[k].R < max_dist \
                        and planes_temp[k].mod < 2:
                    np.append(planes, planes_temp[k].model_fo(dt))
                else:
                    for j in range(0, planes_number):
                        if planes_temp[k].hex == planes[j].hex:
                            planes[j].prev_speed = planes_temp[k].speed
                            planes[j].pr_vert_rate = planes_temp[k].vert_rate

        first += 1
        planes_temp = []
        planes_mod = [[], [], [], []]
        for j in range(0, planes_number):
            for i in [0, 1, 2,3]:
                pm = copy.copy(planes[j])
                planes_temp = np.append(planes_temp, pm)
                pm.model_fo((1.5+(i/2))*dt)
                planes_mod[i] = np.append(planes_mod[i], pm)

        n_t = planes_number
        print('\n','bf optimization:', time.ctime())
        x_opt = T.mpc2()
        print('\n','optimization:', time.ctime(), x_opt)
        if datetime.datetime.today().second >10:
            d = time.gmtime(time.time()+120)
        else:
            d = time.gmtime(time.time()+60)

        if deg_dist([T.azi, T.ele], x_opt[2:4]) > 0.1:
            sa = str('%.2f' % x_opt[2])
            se = str('%.2f' % x_opt[3])
            msg_old = msg
            msg = str(d.tm_year)+'-'+ time.strftime("%m", d)+'-'+ time.strftime("%d", d)+', '+ time.strftime("%H", d)+\
                  ':'+time.strftime('%M', d)+', '+sa+', '+se+', predicted observation\n'
            T.move(msg_old, msg)
            print('mooving:', time.ctime())

        #T.show_cost()
        T.plots(x_opt)

        dist = []
        for pl in planes_mod[1]:
            dist.append(deg_dist([pl.azi,  pl.ele], [x_opt[2], x_opt[3]]))
        m = min(dist)
        ind = np.nanargmin(dist)
        mov = deg_dist([T.azi, T.ele], [x_opt[2], x_opt[3]])
        if mov < 0.1:
            print('No movement from ', T.azi.round(2), T.ele.round(2), ' needed\nClosest plane at:', str('%.2f' % m),
                  'degrees,  in: [', str('%.3f' % planes_mod[1][ind].azi), ',', str('%.3f' % planes_mod[1][ind].ele), ']')
        else:
            print('Moving from:', [T.azi.round(2), T.ele.round(2)], 'to:', x_opt[2:4].round(2), 'dist = ', mov)
            if m < 25:
                print('Caused by planes   in: [', str('%.3f' % planes_mod[1][ind].azi), ',', str('%.3f' % planes_mod[1][ind].ele), ']')
                if m < 3.5:
                    T.store_interference()
            else:
                print('Caused by observation priority: closest plane at:', str('%.2f' % m),' degrees')
        to = time.time()
        time.sleep(dt-(to-ti))
    #except:
    else:
        to = time.time()
        print('no planes')
        time.sleep(dt-(to-ti))

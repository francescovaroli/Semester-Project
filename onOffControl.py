import requests, json, datetime, time, csv
import numpy as np
import logging

dkm = 40000.0 / 360.0
url = 'http://####:8080/aircraft.json'
path = '/home/francesco/PycharmProjects/SemesterProject/data/data20s_20m.csv'


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

    def to_t_coord(self, p):
        dp = p - np.array((self.x, self.y, self.alt))
        dp = dp.reshape((3, 1))
        theta1 = self.azi*np.pi/180
        theta2 = (90-self.ele)*np.pi/180
        c1, s1 = np.cos(theta1), np.sin(theta1)
        c2, s2 = np.cos(theta2), np.sin(theta2)
        r1 = np.matrix([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
        r2 = np.matrix([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
        tr = r1*r2
        return tr.transpose()*dp
    
    def switchOff(self):
        logging.basicConfig(filename="ctrl_output.log", level=logging.INFO, format="%(asctime)-15s  %(message)s")
        logging.info('OFF  ', time.ctime())

    def check_aircraft(self, p):
        p_t = self.to_t_coord(p.pos)
        dist_xy = np.sqrt(p_t[0]**2 + p_t[1]**2)
        return dist_xy > (np.sin(self.beam))*p_t[2]

    
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


T = Telescope(150, 13.5)  
data = Data(url, path)

while True:
    n_pl = data.download()
    for i in range(0, n_pl):
        plane = Plane(path, i)
        r = 1
        if not T.check_aircraft(plane):
            print('\nPlane in Telescope beam:', )
            print('\nplane coordinates:\n', plane.pos)
            print('\ncoordinates in telescope frame:\n', T.to_t_coord(plane.pos))
            r=0
            T.switchOff()
    if r:
        print('No plane in the beam  ', time.ctime())
    time.sleep(2)

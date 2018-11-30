import requests, json, datetime, time, csv

dt =20
url = 'http://129.132.63.205:8080/aircraft.json'
path = '/home/francesco/PycharmProjects/SemesterProject/data/data20s_60m.csv'


class Data:

    def __init__(self, urld, file_path):
        self.url = urld
        self.file_path = file_path

    def downlad(self, i):
        try:
            new_data = json.loads(requests.get(self.url).content)['aircraft']
            now = str(datetime.datetime.now())
            fields = ['idx', 'time', 'hex', 'flight', 'lat', 'lon', 'altitude', 'vert_rate',
                      'track', 'speed', 'rssi', 'seen_pos']
            with open(self.file_path, 'a') as writeFile:
                writer = csv.DictWriter(writeFile, fieldnames=fields)
                for line in new_data:
                    if all(t in line for t in ('lat', 'lon', 'altitude')):
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
                        line['idx'] = i
                        writer.writerow(line)

        except ImportError:
            print("ERROR downloading data")

data = Data(url, path)
t0 = time.time()
t = time.time()
idx = 0

while t-t0 < 60*60:
    data.downlad(idx)
    time.sleep(dt)
    t = time.time()
    idx += 1
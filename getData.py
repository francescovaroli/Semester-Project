#!/usr/bin/env
import requests, json, datetime, time, csv

get_data = lambda: json.loads(requests.get('http://"path"/aircraft.json').content)
data_diff = lambda new, old: [x for x in new if x not in old]
make_date = lambda: str(datetime.datetime.now())

old_data = []

path = '/home/francesco/PycharmProjects/SemesterProject/data/'
period = 0
while True:
        if period > 1000:
            break
        try:
                period += 1
                new_data = get_data()['aircraft']
                diffs = data_diff(new_data, old_data)
                old_data = new_data
                now = make_date()
                if len(diffs) > 0:
                        fields = ['time', 'hex', 'squawk', 'flight', 'lat', 'lon', 'nucp', 'seen_pos', 'altitude','vert_rate', 'track', 'speed', 'category', 'mlat', 'tisb', 'messages', 'seen', 'rssi']
                        with open(path+'dataOUT.csv', 'w') as writeFile:
                                writer = csv.DictWriter(writeFile, fieldnames=fields)
                                writer.writeheader()

                                for line in diffs:
                                        if 'lat' in line:
                                                line['time'] = now
                                                writer.writerow(line)


        except:
                pass
time.sleep(5)


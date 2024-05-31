import json

import h5py as h5
import numpy as np
from picoscenes import Picoscenes
from utils import mac_dec2hex

import matplotlib.pyplot as plt
import os

import torch


class CSI2Dataset:
    def __init__(self, csi_address, mac_list_address):
        self.frames = Picoscenes(csi_address)
        with open(mac_list_address, 'r') as json_file:
            self.mac_list = json.load(json_file)

        self.package_index = [[] for _ in range(len(self.mac_list) - 1)]

        for packet in range(self.frames.count):
            station = self._mac_address_check(self.frames.raw[packet], self.mac_list)
            if station != -1 and self.frames.raw[packet]['RxSBasic']['packetFormat'] == 1:
                self.package_index[station].append(packet)

        print(self.package_index[0] == self.package_index[1])

        print("=" * 30)
        print('Number of Captured Packages:')
        for key, value in self.mac_list.items():

            if key != "AP":
                print(f"{key}({value}): {len(self.package_index[int(key[3])])}")
            else:
                print(f"AP: {value}")

    def show(self, time_range=None):
        print("=" * 30)
        print("Number of Packages in the time range:")
        for i, subindex in enumerate(self.package_index):
            inrange = 0
            for j, index in enumerate(subindex):
                if time_range:
                    if not time_range[0] <= self.frames.raw[index]['RxSBasic']['systemns'] * 10e9 <= time_range[1]:
                        continue
                inrange += 1
            print("STA" + str(i) + ": " + str(inrange))

    def _mac_address_check(self, frame, mac_list):
        stations = list(mac_list.values())[1:]
        if frame['StandardHeader']['ControlField']['FromDS'] == 0:
            if mac_dec2hex(frame["StandardHeader"]['Addr1']) == mac_list['AP'] and \
                    mac_dec2hex(frame["StandardHeader"]['Addr2']) in stations:
                return stations.index(mac_dec2hex(frame["StandardHeader"]['Addr2']))
        return -1

    def save(self, h5_address, group_name, save_limit=None, time_range=None, extra_label=None):
        print("=" * 30)
        saved_count = 0
        with h5.File(h5_address, 'a') as f:
            try:
                group = f[group_name]
            except KeyError:
                print("Group not found, creating a new group...")
                group = f.create_group(group_name)

                complex_dt = h5.vlen_dtype(np.dtype('complex128'))
                group.create_dataset('IQ_Samples', (0,), dtype=complex_dt, maxshape=(None,), compression="gzip")
                group.create_dataset('CSI', (0,), dtype=complex_dt, maxshape=(None,), compression="gzip")
                group.create_dataset('CFO', (0,), dtype=np.dtype('int32'), maxshape=(None,),
                                     compression="gzip")
                group.create_dataset('Timestamp', (0,), dtype=np.dtype('int64'), maxshape=(None,),
                                     compression="gzip")
                group.create_dataset('Station', (0,), dtype=np.dtype('int16'), maxshape=(None,),
                                     compression="gzip")

                if extra_label:
                    extra = group.create_dataset(extra_label[0], (0,), dtype=np.dtype('int16'), maxshape=(None,),
                                                 compression="gzip")

            iq = group['IQ_Samples']
            csi = group['CSI']
            cfo = group['CFO']
            timestamp = group['Timestamp']
            sta = group['Station']

            if extra_label:
                extra = group[extra_label[0]]

            for i, subindex in enumerate(self.package_index):
                saved_count_per_sta = 0
                for j, index in enumerate(subindex):
                    if save_limit:
                        if saved_count_per_sta == save_limit[i]:
                            break
                    if time_range:
                        if not time_range[0] <= self.frames.raw[index]['RxSBasic']['systemns'] * 10e9 <= time_range[1]:
                            continue

                    iq.resize(iq.shape[0] + 1, axis=0)
                    csi.resize(csi.shape[0] + 1, axis=0)
                    cfo.resize(cfo.shape[0] + 1, axis=0)
                    timestamp.resize(timestamp.shape[0] + 1, axis=0)
                    sta.resize(sta.shape[0] + 1, axis=0)

                    iq[iq.shape[0] - 1] = self.frames.raw[index + 100]['BasebandSignals'].flatten()
                    csi[csi.shape[0] - 1] = np.array(self.frames.raw[index + 100]['CSI']['CSI'], dtype=np.complex128)
                    cfo[cfo.shape[0] - 1] = np.int32(self.frames.raw[index + 100]['RxExtraInfo']['cfo'])
                    timestamp[timestamp.shape[0] - 1] = np.int64(self.frames.raw[index + 100]['RxSBasic']['systemns'])
                    sta[sta.shape[0] - 1] = np.int16(i)

                    if extra_label:
                        extra.resize(extra.shape[0] + 1, axis=0)
                        extra[extra.shape[0] - 1] = np.int16(extra_label[1])

                    saved_count += 1
                    saved_count_per_sta += 1
        print(str(saved_count) + ' packages saved!')


class H5Processor:
    def __init__(self, h5_address, group_name):
        with h5.File(h5_address, 'r') as f:
            group = f[group_name]
            stations = group["Station"][:]
            self.station_num = stations.max() + 1
            self.package_index = []

            for i in range(self.station_num):
                self.package_index.append(np.where(stations == i)[0])

            self.cfo = group["CFO"][:]
            self.csi = group["CSI"][:]


def plot_cfo(day0_cfo, day1_cfo, day0_index, day1_index):
    cfo_fig = plt.figure()

    for i in range(len(day0_index)):
        x = np.arange(len(day0_index[i]) + len(day1_index[i]) + 1)
        plt.plot(x, np.hstack((day0_cfo[day0_index[i]], np.array(np.nan), day1_cfo[day1_index[i]])), label=f'STA{i}')

    yticks = np.arange(-30000, 1000, 5000)
    yticklabels = [f"{abs(t) // 1000}k" if t != 0 else "0" for t in yticks]
    plt.yticks(yticks, yticklabels)
    # 设置横轴刻度的标签
    xticks_pos = np.arange(1, 3002, 1501)
    xticks_labels = [f"day{i}" for i in range(2)]
    plt.xticks(xticks_pos, xticks_labels)

    # 添加纵轴单位
    plt.ylabel("kHz")
    plt.legend()
    plt.ylim(-30000, 0)
    plt.savefig('cfo-plot-july20-july21.png')
    plt.show()


def plot_csi(csi, index):
    x = np.arange(-26, 27)

    np.random.seed(42)
    indices = np.random.randint(0, len(index[0]), size=10)
    csi = np.stack(csi, axis=0)

    plt.figure(figsize=(18, 5))

    for sample in range(len(indices)):
        plt.plot(x, np.abs(csi[index[i][indices[sample]]]), label=f'Sample{sample}')

    plt.ylim(0, 1)
    plt.ylabel('Magnitude')
    plt.grid(True)
        # plt.legend()

    plt.ylim(-3.5, 3.5)

    plt.ylabel('Phase (radians)')
    plt.grid(True)
        # plt.legend()
    plt.savefig('plot/csi-plot-july20.png')
    plt.xlabel('Subcarrier Index')
    plt.show()


if __name__ == "__main__":

    for filename in os.listdir("03_08/"):
        file_path = os.path.join("03_08/", filename)
        print(file_path)
        CSI = CSI2Dataset(file_path, "mac_addresses.json")
            # CSI.show()
        CSI.save('dg_dataset.h5', '4', [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
                     extra_label=['Domain', 4])

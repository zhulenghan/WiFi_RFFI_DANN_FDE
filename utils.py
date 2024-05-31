import numpy as np
from scipy.fft import fft, fftshift
from picoscenes import Picoscenes
import os
from matplotlib import pyplot as plt


def mac_dec2hex(mac_address):
    hex_list = [format(num, '02X') for num in mac_address]
    hex_string = ":".join(hex_list)
    return hex_string


def complex_convert_to_2d(data_complex):
    return np.stack((np.real(data_complex), np.imag(data_complex)), axis=0)


def cfo_compensate(preamble, cfo, fs):
    preamble *= np.exp(-1j * 2 * np.pi * np.arange(0, preamble.size) * cfo / fs).flatten()
    return preamble


def preamble_fft_40cbw(preamble, csi=None):
    features = np.stack((fftshift(fft(preamble[64:192])), fftshift(fft(preamble[192:320])),
                         fftshift(fft(preamble[384:512])), fftshift(fft(preamble[512:640])),
                         fftshift(fft(preamble[1152:1280]))), axis=0)
    features = np.concatenate((features[:, 6:63], features[:, 66:123]), axis=1)
    if csi is not None:
        csi_cut = np.concatenate((csi[:58], csi[61:]))

        features = features/csi_cut


    features = complex_convert_to_2d(features)


    #features[:, i, :] /= np.linalg.norm(features[:, i, :], 'fro')
    features /= np.max(np.abs(features))

    return features


def cfo_fde_40cbw(x, cfo, csi):
    return preamble_fft_40cbw(cfo_compensate(x, cfo, 40e6), csi)

def legacy_preable(x, sampling_rate=40):
    if sampling_rate == 40:
        x = x[:640]
        x = complex_convert_to_2d(x)
        x = (x - np.mean(x))/np.std(x)
        return x

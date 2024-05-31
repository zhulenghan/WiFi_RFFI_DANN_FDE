from picoscenes import Picoscenes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import subprocess
import matplotlib.pyplot as plt
import os
import glob
import torch
from models import RFFI
from utils import cfo_fde_40cbw
import numpy as np
import torch.nn.functional as F
import threading
import tkinter as tk

import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import subprocess
import threading
import signal

from dataset_create import mac_dec2hex


def _mac_address_check(frame, mac_list):
    stations = list(mac_list.values())[1:]
    if frame['StandardHeader']['ControlField']['FromDS'] == 0:
        if mac_dec2hex(frame["StandardHeader"]['Addr1']) == mac_list['AP'] and \
                mac_dec2hex(frame["StandardHeader"]['Addr2']) in stations:
            return stations.index(mac_dec2hex(frame["StandardHeader"]['Addr2']))
    return -1


class CaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Capture Output")

        # Label and Entry for model file path
        self.model_label = tk.Label(root, text="Model (.pth):")
        self.model_label.pack(fill=tk.X, padx=10)

        self.model_entry = tk.Entry(root)
        self.model_entry.pack(fill=tk.X, padx=10, pady=5)

        # Configure layout using PanedWindow or Frames
        self.main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_pane.pack(fill=tk.BOTH, expand=True)

        # Adjust the ScrolledText size and add it to the left pane
        self.output_text = ScrolledText(self.main_pane, height=10, width=50)  # Adjust size as needed
        self.main_pane.add(self.output_text)

        # Right pane for label and Matplotlib plot
        self.right_pane = tk.Frame(self.main_pane)
        self.main_pane.add(self.right_pane)

        # Label for "This is STAn"
        self.sta_label = tk.Label(self.right_pane, text="This is STA?")
        self.sta_label.pack()

        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.plot = self.fig.add_subplot(111)
        self.plot_widget = FigureCanvasTkAgg(self.fig, self.right_pane)
        self.plot_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.start_button = tk.Button(root, text="Start Capture", command=self.start_capture_thread)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=10)

        self.stop_button = tk.Button(root, text="Stop Capture", command=self.stop_capture)
        self.stop_button.pack(side=tk.RIGHT, padx=5, pady=10)

        self.process = None
        self.additional_thread_running = False

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_plot(self, new_data):
        # Assuming new_data is a tuple or list of (x, y) points to plot
        self.plot.clear()  # Clear existing plot
        x_positions = range(8)  # [0, 1, 2, 3, 4, 5, 6, 7]
        labels = ['STA0', 'STA1', 'STA2', 'STA3', 'STA4', 'STA5', 'STA6', 'STA7']

        # Draw new bar chart
        self.plot.bar(x_positions, new_data, tick_label=labels, color='blue')

        # Optionally customize the plot further, e.g., setting a title
        self.plot.set_title("Probabilities")

        self.plot_widget.draw()

    def update_sta_label(self, n):
        new_text = f"This is STA{n}"
        self.sta_label.config(text=new_text)

    def run_capture(self):
        # Start the subprocess as the leader of a new process group
        self.process = subprocess.Popen(["bash", "capture.sh"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, preexec_fn=os.setsid)
        for line in self.process.stdout:
            self.update_text(line)
        if self.process:
            self.process.wait()

    def additional_thread_task(self, model_path):
        print(f"Model: {model_path}")
        count = 0
        while self.additional_thread_running:
            csi_files = glob.glob('*.csi')
            if csi_files:  # If there are any .csi files
                # Assuming you want to work with the first found .csi file
                csi_file_name = csi_files[0]
                model.load_state_dict(torch.load(model_path))
                break

        while self.additional_thread_running:
            frames = Picoscenes(csi_file_name)
            # Here you can do additional tasks, for now, just sleep for a bit
            # Check for .csi files in the current working directory
            if frames.count > count:
                count = frames.count
                frame = frames.raw[count - 1]
                station = _mac_address_check(frame, mac)
                if station != -1 and frame['RxSBasic']['packetFormat'] == 1:
                    self.root.after(0, self.update_sta_label, station)
                    x = frame['BasebandSignals'].flatten()
                    cfo = np.int32(frame['RxExtraInfo']['cfo'])
                    csi = np.array(frame['CSI']['CSI'], dtype=np.complex128)
                    x = cfo_fde_40cbw(x, cfo, csi)
                    x = torch.from_numpy(x).float().cuda().unsqueeze(0)

                    pred = F.softmax(model(x), dim=1).flatten().cpu().detach().numpy()

                    self.update_plot(pred)


    def start_capture_thread(self):
        if self.process and self.process.poll() is None:
            return  # Process is still running, do nothing

        model_path = self.model_entry.get()  # Get model file path from entry

        # Indicate that the additional thread should be running
        self.additional_thread_running = True





        # Start the capture process thread
        capture_thread = threading.Thread(target=self.run_capture)
        capture_thread.daemon = True
        capture_thread.start()

        # Start the additional thread with model_path as an argument
        additional_thread = threading.Thread(target=self.additional_thread_task, args=(model_path,))
        additional_thread.daemon = True
        additional_thread.start()

    def update_text(self, text):
        if self.output_text:
            self.output_text.insert(tk.END, text)
            self.output_text.see(tk.END)
            self.root.update_idletasks()

    def stop_capture(self):
        # Terminate the entire process group
        if self.process and self.process.poll() is None:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait()
            self.update_text("\nCapture stopped.\n")

        # Indicate that the additional thread should stop
        self.additional_thread_running = False

        csi_files = glob.glob('*.csi')

        # Loop through the list of .csi files and delete each one
        for file in csi_files:
            os.remove(file)
            self.update_text(f"Deleted {file}")

    def on_closing(self):
        self.stop_capture()
        self.root.destroy()


if __name__ == "__main__":
    os.chdir("demo")
    mac = {
        "AP": "5C:02:14:01:AF:D9",
        "STA0": "EC:60:73:C6:B8:C7",
        "STA1": "EC:60:73:C6:B8:C5",
        "STA2": "F4:84:8D:2B:5F:56",
        "STA3": "EC:60:73:DC:12:8E",
        "STA4": "F4:84:8D:2B:5F:0C",
        "STA5": "EC:60:73:C6:D9:3C",
        "STA6": "EC:60:73:DC:17:2D",
        "STA7": "F4:84:8D:2B:3A:C3"
    }

    model = RFFI(num_classes=8, l2_norm=True).cuda()
    model.eval()

    root = tk.Tk()
    app = CaptureApp(root)
    root.mainloop()

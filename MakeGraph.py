from time import sleep

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PlaySound

import StatusCheck


class MakeGraph:
    def __init__(self, root, sc):
        self.sc = sc
        self.fig = plt.figure(figsize=(10, 3))  # figure(도표) 생성
        max_points = 50
        self.ax = plt.subplot(211, xlim=(0, max_points), ylim=(0, 100))
        self.line, = self.ax.plot(np.arange(max_points),
                                  np.ones(max_points, dtype=np.float) * np.nan, lw=1, c='blue', ms=100)
        old_y = self.line.get_ydata()
        new_y = np.r_[old_y[1:], 80]
        self.line.set_ydata(new_y)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)  #
        self.canvas.get_tk_widget().pack(side="bottom")  #

    # private
    def __get_line(self):
        return self.line

    # private
    def __animate(self, i):
        old_y = self.line.get_ydata()
        self.score = old_y[-1] + StatusCheck.StatusCheck.check_weight[self.sc.check()]
        if self.score > 99:
            self.score = 99
        elif self.score < 1:
            self.score = 1

        new_y = np.r_[old_y[1:], self.score]
        self.line.set_ydata(new_y)
        print(new_y)
        if self.score <= 50:
            PlaySound.PlaySound.play("sound/sound1.wav")
        sleep(0.5)
        return self.line

    def start_graph(self):
        anim = animation.FuncAnimation(self.fig, self.__animate, init_func=self.__get_line, frames=60, interval=1000,
                                       blit=False)
        sleep(0.1)

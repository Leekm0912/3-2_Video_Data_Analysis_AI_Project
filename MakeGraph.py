from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import random
import time
#
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as Tk


class MakeGraph:
    def __init__(self, root):
        self.fig = plt.figure()  # figure(도표) 생성
        max_points = 50
        self.ax = plt.subplot(211, xlim=(0, max_points), ylim=(0, 100))
        self.line, = self.ax.plot(np.arange(max_points),
                                  np.ones(max_points, dtype=np.float) * np.nan, lw=1, c='blue', ms=1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)  #
        self.canvas.get_tk_widget().pack(side="bottom")  #

        anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.getLine, frames=200, interval=50,
                                       blit=False)

    def getLine(self):
        return self.line

    def animate(self, i):
        y = random.randint(0, 100)
        old_y = self.line.get_ydata()
        new_y = np.r_[old_y[1:], y]
        self.line.set_ydata(new_y)
        print(new_y)
        return self.line


if __name__ == '__main__':
    Tk.mainloop()

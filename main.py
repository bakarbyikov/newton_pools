from random import choices
from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy.polynomial import Polynomial as Poly


# We just subclass Rectangle so that it can be called with an Axes
# instance, causing the rectangle to update its shape to match the
# bounds of the Axes
class UpdatingRect(Rectangle):
    def __call__(self, ax):
        self.set_bounds(*ax.viewLim.bounds)
        ax.figure.canvas.draw_idle()


# A class that will regenerate a fractal set as we zoom in, so that you
# can actually see the increasing detail.  A box in the left panel will show
# the area to which we are zoomed.
class MandelbrotDisplay:
    def __init__(self, h=500, w=500, niter=500, eps=1e-5):
        polynominal = Poly([-6, 11, -6, 1])
        # polynominal = Poly(choices(range(-10, 10), k=5))
        print(polynominal)
        self.poly = polynominal
        self.deriv = polynominal.deriv()
        self.height = h
        self.width = w
        self.niter = niter
        self.eps = eps

    def compute_image(self, xstart, xend, ystart, yend):
        print(f"computing {self.width*self.height} pixels")
        self.x = np.linspace(xstart, xend, self.width)
        self.y = np.linspace(ystart, yend, self.height).reshape(-1, 1)
        threshold_time = np.zeros((self.height, self.width))
        z = np.empty(threshold_time.shape, dtype=complex)
        z.real = self.x
        z.imag  = self.y
        mask = np.ones(threshold_time.shape, dtype=bool)
        then = perf_counter()
        for i in range(self.niter):
            f = self.poly(z[mask])
            f_d = self.deriv(z[mask])
            delta = f/f_d
            z[mask] -= delta
            mask[mask] &= (np.abs(delta) > self.eps)
            if not mask.any():
                break
            # mask[mask] &= (np.abs(f) > self.eps)
            threshold_time += mask
        elapsed = perf_counter() - then
        print(f"{elapsed = }")
        pps = self.width * self.height / elapsed
        print(f"{pps = }")
        return threshold_time

    def ax_update(self, ax):
        ax.set_autoscale_on(False)  # Otherwise, infinite loop
        # Get the number of points from the number of pixels in the window
        self.width, self.height = \
            np.round(ax.patch.get_window_extent().size).astype(int)
        # Get the range for the new area
        vl = ax.viewLim
        extent = vl.x0, vl.x1, vl.y0, vl.y1
        # Update the image object with our new data and extent
        im = ax.images[-1]
        im.set_data(self.compute_image(*extent))
        im.set_extent(extent)
        ax.figure.canvas.draw_idle()


md = MandelbrotDisplay()
Z = md.compute_image(-20., 20., -20., 20.)

fig1, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(Z, origin='lower', cmap='twilight',
           extent=(md.x.min(), md.x.max(), md.y.min(), md.y.max()))
ax2.imshow(Z, origin='lower', cmap='twilight',
           extent=(md.x.min(), md.x.max(), md.y.min(), md.y.max()))

rect = UpdatingRect(
    [0, 0], 0, 0, facecolor='none', edgecolor='black', linewidth=1.0)
rect.set_bounds(*ax2.viewLim.bounds)
ax1.add_patch(rect)

# Connect for changing the view limits
ax2.callbacks.connect('xlim_changed', rect)
ax2.callbacks.connect('ylim_changed', rect)

ax2.callbacks.connect('xlim_changed', md.ax_update)
ax2.callbacks.connect('ylim_changed', md.ax_update)
ax2.set_title("Zoom here")

plt.show()
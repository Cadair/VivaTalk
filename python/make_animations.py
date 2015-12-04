# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:59:23 2015

@author: stuart
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.animation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from streamlines import Streamlines

#Use Equation 1 to calculate the vector field in a 2D plane to plot it.
frames = 250
tmax = 241.
time = np.linspace(0,tmax,frames)
dt = time[1:] - time [:-1]
period = 240.

x = np.linspace(7812.5,1992187.5,128)
y = np.linspace(7812.5,1992187.5,128)

x_max = x.max()
y_max = y.max()

xc = 1.0e6
yc = 1.0e6

xn = x - xc
yn = y - yc

delta_x=0.1e6
delta_y=0.1e6

xx, yy = np.meshgrid(xn,yn)
exp_y = np.exp(-(yn**2.0/delta_y**2.0))
exp_x = np.exp(-(xn**2.0/delta_x**2.0))

exp_x2, exp_y2= np.meshgrid(exp_x,exp_y)
exp_xyz = exp_x2 * exp_y2


#==============================================================================
# Define Driver Equations and Parameters
#==============================================================================
#A is the amplitude, B is the spiral expansion factor
A = 10

#Tdamp defines the damping of the driver with time, Tdep is the ocillator
tdamp = lambda time1: 1.0 #*np.exp(-(time1/(period)))
tdep = lambda time1: np.sin(2.0*np.pi*(time1/period)) * tdamp(time1)

#Define a peak index to use for scaling in the inital frame
max_ind = np.argmax(tdep(time))
print "max:", max_ind
print time
print tdep(time)

def log():
    B = 0.05
    phi = np.arctan2(1,B)
    theta = np.arctan2(yy,xx)

    uy = np.sin(theta + phi)
    ux =  np.cos(theta + phi)

    vx = lambda time1: (ux / np.sqrt(ux**2 + uy**2)) * exp_xyz * tdep(time1) * A
    vy = lambda time1: (uy / np.sqrt(ux**2 + uy**2)) * exp_xyz * tdep(time1) * A

    vv = np.sqrt(vx(time[max_ind])**2 + vy(time[max_ind])**2)

    return vx, vy, vv

def arch():
    B = 0.005
    r = np.sqrt(xx**2 + yy**2)

    vx = lambda time1: ( (B*1e6 * xx) / (xx**2 + yy**2) + yy/r ) * exp_xyz * tdep(time1) * A
    vy = lambda time1: ( (B*1e6 * yy) / (xx**2 + yy**2) - xx/r ) * exp_xyz * tdep(time1) * A

    vv = np.sqrt(vx(time[max_ind])**2 + vy(time[max_ind])**2)

    return vx, vy, vv

def uniform():
    #Uniform
    vx = lambda time1: A * (yy / np.sqrt(xx**2 + yy**2)) * exp_xyz * tdep(time1)
    vy = lambda time1: A * (-xx / np.sqrt(xx**2 + yy**2)) * exp_xyz * tdep(time1)
    vv = np.sqrt(vx(time[max_ind])**2 + vy(time[max_ind])**2)

    return vx, vy, vv


drivers = [log, arch, uniform]


def animate_plot(i):
    global im, qu, time, x, y, driver_func
    print(i)
    vx, vy, vv = driver_func()
    im.set_data(np.sqrt(vx(time[i])**2 + vy(time[i])**2))

    qu.set_UVC(vx(time[i]), vy(time[i]))



for driver_func in drivers:
    fig, ax = plt.subplots(figsize=(7,6), dpi=300)
    #============================================================================
    # Do the Plotting
    #============================================================================
    vx, vy, vv = driver_func()
    # Calculate Streamline
    slines = Streamlines(x,y,vx(time[max_ind]),vy(time[max_ind]),maxLen=7000,
                         x0=xc, y0=yc, direction='forwards')

    im = ax.imshow(vv, cmap='Blues', extent=[7812.5,x_max,7812.5,y_max])
    im.set_norm(matplotlib.colors.Normalize(vmin=0,vmax=A))
    #ax.hold()

    if driver_func != uniform:
        Sline, = ax.plot(slines.streamlines[0][0],slines.streamlines[0][1],color='red',linewidth=2, zorder=40)
    else:
        Sline = matplotlib.patches.Circle([1e6, 1e6], radius=.15e6, fill=False, color='red', linewidth=2, zorder=40)
        ax.add_artist(Sline)

    #Add colourbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = plt.colorbar(im,cax)
    cbar.set_label(r"$|V|$ [ms$^{-1}$]")
    scalar = matplotlib.ticker.ScalarFormatter(useMathText=False,useOffset=False)
    scalar.set_powerlimits((-3,3))
    cbar.formatter = scalar
    cbar.ax.yaxis.get_offset_text().set_visible(True)
    cbar.update_ticks()
    #cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")

    #Add quiver plot overlay
    qu = ax.quiver(x, y, vx(time[max_ind]), vy(time[max_ind]), scale=25*A, color='k', zorder=20, linewidth=1)
    ax.axis([8.0e5,12.0e5,8.0e5,12.0e5])

    ax.xaxis.set_major_formatter(scalar)
    ax.yaxis.set_major_formatter(scalar)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)
    ax.set_xlabel("X [Mm]")
    ax.set_ylabel("Y [Mm]")

    ani = matplotlib.animation.FuncAnimation(fig, animate_plot, frames=frames)
    ani.save('images/driver_{}.mp4'.format(driver_func.__name__), savefig_kwargs={'transparent': True}, fps=frames/15., dpi=600, bitrate=10000)

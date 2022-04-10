import os
import matplotlib.pyplot as plt
import pandas as pd
import random as rand

from math import radians
from matplotlib.animation import FuncAnimation
from libs.stanley_controller import StanleyController
from libs.car_description import Description
from libs.cubic_spline_interpolator import generate_cubic_spline
from kinematic_model import KinematicBicycleModel

class Simulation:

    def __init__(self):

        fps = 50.0

        self.dt = 1/fps
        self.map_size_x = 70
        self.map_size_y = 40
        self.frames = 2500
        self.loop = False

class Path:

    def __init__(self):

        # Get path to waypoints.csv
        dir_path = 'data/waypoints.csv'
        df = pd.read_csv(dir_path)

        x = df['X-axis'].values
        y = df['Y-axis'].values
        ds = 0.05

        self.px, self.py, self.pyaw, _ = generate_cubic_spline(x, y, ds)

class Car:

    def __init__(self, init_x, init_y, init_yaw, px, py, pyaw, dt):

        # Model parameters
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.v = 0.0
        self.delta = 0.0
        self.omega = 0.0
        self.wheelbase = 2.96
        self.max_steer = radians(33)
        self.dt = dt
        self.c_r = 0.01
        self.c_a = 2.0

        # Tracker parameters
        self.px = px
        self.py = py
        self.pyaw = pyaw
        self.k = 8.0
        self.ksoft = 1.0
        self.kyaw = 0.01
        self.ksteer = 0.0
        self.crosstrack_error = None
        self.target_id = None

        # Description parameters
        self.overall_length = 1.97
        self.overall_width = 0.5
        self.tyre_diameter = 0.66
        self.tyre_width = 0.032
        self.axle_track = 1.662
        self.rear_overhang = (self.overall_length - self.wheelbase) / 2
        self.colour = 'black'

        self.tracker = StanleyController(self.k, self.ksoft, self.kyaw, self.ksteer, self.max_steer, self.wheelbase, self.px, self.py, self.pyaw)
        self.kbm = KinematicBicycleModel(self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a)

    def drive(self):
        
        throttle = rand.uniform(150, 200)
        self.delta, self.target_id, self.crosstrack_error = self.tracker.stanley_control(self.x, self.y, self.yaw, self.v, self.delta)
        self.x, self.y, self.yaw, self.v, _, _ = self.kbm.kinematic_model(self.x, self.y, self.yaw, self.v, throttle, self.delta)

        os.system('cls' if os.name=='nt' else 'clear')
        print(f"Cross-track term: {self.crosstrack_error}")

def init_anim(): pass
def animate(frame, *args):

    ax, sim, path, car, desc, outline, fr, rr, fl, rl, rear_axle, annotation, target = args

    # Camera tracks car
    ax.set_xlim(car.x - sim.map_size_x, car.x + sim.map_size_x)
    ax.set_ylim(car.y - sim.map_size_y, car.y + sim.map_size_y)

    # Drive and draw car
    car.drive()
    outline_plot, fr_plot, rr_plot, fl_plot, rl_plot = desc.plot_car(car.x, car.y, car.yaw, car.delta)
    outline.set_data(*outline_plot)
    fr.set_data(*fr_plot)
    rr.set_data(*rr_plot)
    fl.set_data(*fl_plot)
    rl.set_data(*rl_plot)
    rear_axle.set_data(car.x, car.y)

    # Show car's target
    target.set_data(path.px[car.target_id], path.py[car.target_id])

    # Annotate car's coordinate above car
    annotation.set_text(f'{car.x:.1f}, {car.y:.1f}')
    annotation.set_position((car.x, car.y + 5))

    plt.title(f'{sim.dt*frame:.2f}s', loc='right')
    plt.xlabel(f'Speed: {car.v:.2f} m/s', loc='left')
    # plt.savefig(f'image/visualisation_{frame:03}.png', dpi=300)

    return outline, fr, rr, fl, rl, rear_axle, target,

def main():
    
    sim = Simulation()
    path = Path()
    car = Car(path.px[0], path.py[0], path.pyaw[0], path.px, path.py, path.pyaw, sim.dt)
    desc = Description(car.overall_length, car.overall_width, car.rear_overhang, car.tyre_diameter, car.tyre_width, car.axle_track, car.wheelbase)

    interval = sim.dt * 10**3

    fig = plt.figure()
    ax = plt.axes()
    ax.set_aspect('equal')

    road = plt.Circle((0, 0), 50, color='gray', fill=False, linewidth=30)
    ax.add_patch(road)
    ax.plot(path.px, path.py, '--', color='gold')

    annotation = ax.annotate(f'{car.x:.1f}, {car.y:.1f}', xy=(car.x, car.y + 5), color='black', annotation_clip=False)
    target, = ax.plot([], [], '+r')

    outline, = ax.plot([], [], color=car.colour)
    fr, = ax.plot([], [], color=car.colour)
    rr, = ax.plot([], [], color=car.colour)
    fl, = ax.plot([], [], color=car.colour)
    rl, = ax.plot([], [], color=car.colour)
    rear_axle, = ax.plot(car.x, car.y, '+', color=car.colour, markersize=2)

    fargs = (ax, sim, path, car, desc, outline, fr, rr, fl, rl, rear_axle, annotation, target)
    
    _ = FuncAnimation(fig, animate, frames=sim.frames, init_func=init_anim, fargs=fargs, interval=interval, repeat=sim.loop)
    # anim.save('animation.gif', writer='imagemagick', fps=50)
    
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()

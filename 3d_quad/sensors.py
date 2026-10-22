import numpy as np
import math
from scipy.spatial.transform import Rotation as R

from datetime import datetime
import ppigrf

class Magnetometer:
    def __init__(self, lon, lat, h):
        self.lon = lon
        self.lat = lat
        self.h = h
        self.B = self.read()
        self.B_rot = np.zeros(3)
    

    def read(self):
        date = datetime.now()
        Be, Bn, Bu = ppigrf.igrf(self.lon, self.lat, self.h, date)
        return np.array([Bn[0][0],  Be[0][0], -Bu[0][0]]) / 1e5 # convert nT to gauss

    def update(self, q):
        self.B_rot =  R.from_quat(q).inv().apply(self.B) + np.random.normal(0, 0.02, 3)


    @property
    def field(self):
        return self.B_rot


class GPS:
    def __init__(self, lon, lat, h):
        self.lon = lon
        self.lat = lat
        self.h = h

    def read(self):
        # In a real-world scenario, you would read the current position from the GPS sensor here.
        # In this simulation, we'll just return the initial position with some random noise.
        lon = self.lon + np.random.normal(0, 0.0001)
        lat = self.lat + np.random.normal(0, 0.0001)
        h = self.h + np.random.normal(0, 0.1)
        return lon, lat, h

class IMU:
    def __init__(self) -> None:
        # self.g = np.array([0, 0, -9.81])
        self.acc = np.zeros(3)
        self.gyro = np.zeros(3)
        self.time = 0
    
    def update(self, acc, gyro): # add gaussian noise
        #      The sensor signals reconstruction and noise levels are from [1] (SEE SIHSIM IN PX4) 
        # [1] Bulka, Eitan, and Meyer Nahon. "Autonomous fixed-wing aerobatics: from theory to flight."
        #     In 2018 IEEE International Conference on Robotics and Automation (ICRA), pp. 6573-6580. IEEE, 2018.
        std_acc = np.array([0.5, 1.7, 1.4])
        std_gyro = np.array([0.14, 0.07, 0.03])
        self.acc = acc + np.random.normal(0, 1, 3) * std_acc
        self.gyro = gyro + np.random.normal(0, 1, 3) * std_gyro

    @property
    def a(self):
        return self.acc
    
    @property
    def w(self):
        return self.gyro
    



class Barometer:
    def __init__(self):
        self._last_update_time = 0
        self._baro_rnd_use_last = False
        self._baro_rnd_y2 = 0
        # self._baro_drift_pa = 0
        # self._baro_drift_pa_per_sec = 0
        # self._sim_baro_off_p = 0
        # self._sim_baro_off_t = 0
        self.pressure = 0
        self.temperature = 0


    def update(self, altitude):
        lapse_rate = 0.0065
        temperature_msl = 288.0

        temperature_local = temperature_msl - lapse_rate * altitude
        pressure_ratio = pow(temperature_msl / temperature_local, 5.256)
        pressure_msl = 101325.0
        absolute_pressure = pressure_msl / pressure_ratio

        y1 = 0
        if not self._baro_rnd_use_last:
            w = 1.0
            while w >= 1.0:
                x1 = 2.0 * np.random.normal() - 1.0
                x2 = 2.0 * np.random.normal() - 1.0
                w = x1 * x1 + x2 * x2

            w = math.sqrt((-2.0 * math.log(w)) / w)
            y1 = x1 * w
            self._baro_rnd_y2 = x2 * w
            self._baro_rnd_use_last = True
        else:
            y1 = self._baro_rnd_y2
            self._baro_rnd_use_last = False

        abs_pressure_noise = 1.0 * y1
        self.pressure = absolute_pressure + abs_pressure_noise 
        self.temperature = temperature_local - 273.0 
      

    @property
    def P(self):
        return self.pressure

    @property
    def T(self):
        return self.temperature
    


if __name__ == "__main__":
       # create a magnetometer sensor at a specific location
    magnetometer = Magnetometer(5.32415, 60.39299, 0)


    # read the magnetic field
    Be, Bn, Bu = magnetometer.read()
    print(f"Be: {Be}, Bn: {Bn}, Bu: {Bu}")

    # calc field in neighboring poi
    # init quaternion with scipy 
    q = R.from_euler('xyz', [0, 0, 0]).as_quat()  

    # measure the speed of get_field function
    # import time
    # start = time.time()
    # for i in range(100):
    #     magnetometer.update(q)
    # end = time.time()
    # # calcualte the number of reads per second
    # print(f"Reads per second: {1000 / (end - start)}")


    # quit()


import time
import tqdm
import os
import threading



class Simulation:
    def __init__(self, vehicle, px4_connector, logger=None) -> None:
        self.vehicle = vehicle
        self.px4 = px4_connector
        self.logger = logger

        self.t_abs_s = time.time()
        self.t_abs_us = round(self.t_abs_s * 1e6)
        self.t_boot_us = round(self.t_abs_s * 1e6)

        # simulation params
        self.sim_time = 10
        self.current_time = 0
        self.set_dt(0.04)

        # tmp for debugging
        self.print_time = 0

        self.heartbeat_timer = None
        self.system_time_timer = None

        self.speed_factor = None

        self.freewheeling = True  # period when simulation runs before px4 starts sending commands
        # if self.px4 is not None:
        #     self.send_heartbeat_periodically()
        #     self.send_system_time_periodically()



    def send_heartbeat_periodically(self):
        self.px4.send_heartbeat()
        # schedule the function to run again after 1 second (1000 milliseconds)
        self.heartbeat_timer = threading.Timer(0.9, self.send_heartbeat_periodically)
        self.heartbeat_timer.start()

    def send_system_time_periodically(self):
        self.px4.send_system_time(self.t_abs_us, self.t_boot_us)
        self.system_time_timer = threading.Timer(4, self.send_system_time_periodically)
        self.system_time_timer.start()

    def set_dt(self, dt):
        self.dt = dt
        self.vehicle.set_dt(dt)
        self.vehicle.init_solver(self.vehicle.dynamics, type='rk4')
   
    def set_sim_time(self, sim_time):
        self.sim_time = sim_time

    def start(self,):
        self.start_time = time.time()
        if self.px4 is None:
            print("px4 is not connected. Performing simulation without px4")
            num_steps = int(self.sim_time / self.dt)
            for i in tqdm.trange(num_steps):
                    # print('step:', i)
                    self.vehicle.dynamics_step()
                    if self.logger is not None:
                        self.logger.record_data(i * self.dt, self.vehicle)
                    self.t_abs_us  += self.dt * 1e6

        elif self.px4.connected:
            try:
                while True:
                    loop_start_time = time.time()
                    # print(self.freewheeling, ' freewheeling')
                    # print every 1 second
                    # if self.t_abs_us % 1000000 == 0:
                        # print("t_abs_us:", self.t_abs_us)
                        # print("t_boot_us:", self.t_boot_us)
                        # print("t_us: ", self.t_us)

                    self.t_us = self.t_abs_us - self.t_boot_us
                    if self.freewheeling:
                        # waiting px4 to boot
                        self.px4.receive(blocking = False)
                        if self.freewheeling and self.px4.received_controls:
                            self.freewheeling = False
                            print("freewheeling finished")
                            continue

                        # self.vehicle.update_controls(self.px4.controls)
                        # self.vehicle.dynamics_step()
                        self.vehicle.IMU.update(self.vehicle.v_d, self.vehicle.w)
                        
                        # self.px4.send_hil_state_quaternion(self.t_abs_us, self.vehicle)
                        self.px4.send_hil_sensor(self.t_abs_us, self.vehicle)

                        # if self.logger is not None:
                            # self.logger.record_data(self.t_abs_us, self.vehicle)
                        #time sleep 
                        # time.sleep(0.01)

                    elif not self.freewheeling: 
                        self.vehicle.update_controls(self.px4.controls)
                        self.vehicle.dynamics_step()
                        self.vehicle.IMU.update(self.vehicle.v_d, self.vehicle.w)
                        self.vehicle.barometer.update(self.vehicle.p[2])
                        self.vehicle.magnetometer.update(self.vehicle.q)
                        
                        # print(self.vehicle.barometer.pressure, self.vehicle.barometer.temperature) 
                        self.px4.send_hil_state_quaternion(self.t_abs_us, self.vehicle)
                        self.px4.send_hil_sensor(self.t_abs_us, self.vehicle)
                        self.px4.send_hil_gps(self.t_abs_us, self.vehicle)
                        # lockstep. wait for controls from px4
                        self.px4.receive(blocking = True)
                        # self.px4.receive(blocking = False)

                    self.print_time += self.dt
                    if self.print_time > 1:
                        print(self.t_us * 1e-6, "s")
                        self.print_time = 0
                        
                    self.t_abs_us  += int(self.dt * 1e6) 
                    loop_end_time = time.time()
                    sleep_time = self.dt / self.speed_factor - (loop_end_time - loop_start_time)
                    if sleep_time > 0 and self.speed_factor is not None:
                        time.sleep(sleep_time)
                   
            except KeyboardInterrupt:
                print("KeyboardInterrupt")
                self.px4.close()
                if self.logger is not None:
                    self.logger.close_logger()

        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time
        print("Simulation time:", self.sim_time, "seconds")
        print("Time step:", self.dt, "seconds")
        print("Execution time:", self.execution_time, "seconds")
        print("real time factor:", self.sim_time/self.execution_time)

    def close(self):
        print("Closing simulation")
        self.px4.close()
        if self.logger is not None:
            self.logger.close_logger()

        if self.heartbeat_timer is not None:
            self.heartbeat_timer.cancel()
        if self.system_time_timer is not None:
            self.system_time_timer.cancel()


    def plot_data(self, plotjugler_path):
        log_path = os.path.join(os.getcwd(), self.logger.file_path)
        xml_schema_path = './plot_layout.xml'
        # have to remove a tag of last opened files (bug in plotjuggler)
        python_script_xmlprunner_path = "./remove_tag.py "
        os.system(f"python3 {python_script_xmlprunner_path} {xml_schema_path}")
        os.system(f"{plotjugler_path} -n -d {log_path} -l {xml_schema_path}")

    def plot_data_windows(self, plotjugler_path):
        log_path = os.path.join(os.getcwd(), self.logger.file_path)
        xml_schema_path = os.path.join(os.getcwd(), 'plot_layout.xml')
        # have to remove a tag of last opened files (bug in plotjuggler)
        python_script_xmlprunner_path = os.path.join(os.getcwd(), 'remove_tag.py')
        os.system(f"python \"{python_script_xmlprunner_path}\" \"{xml_schema_path}\"")
        os.system(f"\"\"{plotjugler_path}\" -n -d \"{log_path}\" -l \"{xml_schema_path}\"\"")

       
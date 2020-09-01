# Bluesky hardware flyer for DE optimization

import time as ttime

from collections import deque

from ophyd.sim import NullStatus


class BlueskyFlyer:
    def __init__(self):
        self.name = 'bluesky_flyer'
        self._asset_docs_cache = deque()
        self._resource_uids = []
        self._datum_counter = None
        self._datum_ids = []

    def kickoff(self):
        return NullStatus()

    def complete(self):
        return NullStatus()

    def describe_collect(self):
        return {self.name: {}}

    def collect(self):
        now = ttime.time()
        data = {}
        yield {'data': data,
               'timestamps': {key: now for key in data},
               'time': now,
               'filled': {key: False for key in data}}

    def collect_asset_docs(self):
        items = list(self._asset_docs_cache)
        self._asset_docs_cache.clear()
        for item in items:
            yield item


class HardwareFlyer(BlueskyFlyer):
    def __init__(self, params_to_change, velocities, time_to_travel,
                 detector, motors, start_det, read_det, stop_det,
                 watch_func):
        super().__init__()
        self.name = 'hardware_flyer'
        # TODO: These 3 lists to be merged later
        self.params_to_change = params_to_change  # dict of dicts; {motor_name: {'position':...}}
        self.velocities = velocities  # dictionary with motor names as keys
        self.time_to_travel = time_to_travel  # dictionary with motor names as keys
        self.detector = detector
        self.motors = motors
        self.watch_positions = {name: {'position': []} for name in self.motors}
        self.watch_intensities = []
        self.watch_timestamps = []
        self.motor_move_status = None
        self.start_det = start_det
        self.read_det = read_det
        self.stop_det = stop_det
        self.watch_func = watch_func

    def kickoff(self):
        slowest_motor = sorted(self.time_to_travel,
                               key=lambda x: self.time_to_travel[x],
                               reverse=True)[0]
        self.start_det(self.detector)
        ttime.sleep(1.0)
        for motor_name, field in self.motors.items():
            for field_name, motor_obj in field.items():
                motor_obj.velocity.put(self.velocities[motor_name])
        for motor_name, field in self.motors.items():
            for field_name, motor_obj in field.items():
                if motor_name == slowest_motor:
                    self.motor_move_status = motor_obj.set(self.params_to_change[motor_name][field_name])
                else:
                    motor_obj.set(self.params_to_change[motor_name][field_name])
        # Call this function once before we start moving all motors to collect the first points.
        self._watch_function()
        self.motor_move_status.watch(self._watch_function)
        return NullStatus()

    def complete(self):
        return self.motor_move_status

    def describe_collect(self):
        return_dict = {self.name:
                       {f'{self.name}_intensity':
                        {'source': f'{self.name}_intensity',
                         'dtype': 'number',
                         'shape': []},
                        }
                       }
        motor_dict = {}
        for motor_name in self.motors.keys():
            motor_dict[f'{self.name}_{motor_name}_velocity'] = {'source': f'{self.name}_{motor_name}_velocity',
                                                                'dtype': 'number', 'shape': []}
            motor_dict[f'{self.name}_{motor_name}_position'] = {'source': f'{self.name}_{motor_name}_position',
                                                                'dtype': 'number', 'shape': []}
        return_dict[self.name].update(motor_dict)
        return return_dict

    def collect(self):
        self.stop_det(self.detector)
        for ind in range(len(self.watch_intensities)):
            motor_dict = {}
            for motor_name, field in self.motors.items():
                for field_name, motor_obj in field.items():
                    motor_dict.update(
                        {f'{self.name}_{motor_name}_velocity': self.velocities[motor_name],
                         f'{self.name}_{motor_name}_position': self.watch_positions[motor_name][field_name][ind]}
                    )
            data = {f'{self.name}_intensity': self.watch_intensities[ind]}
            data.update(motor_dict)
            yield {'data': data,
                   'timestamps': {key: self.watch_timestamps[ind] for key in data},
                   'time': self.watch_timestamps[ind],
                   'filled': {key: False for key in data}}

        # # This will produce one event with dictionaries in the <...>_parameters field.
        # motor_params_dict = {}
        # for motor_name, motor_obj in self.motors.items():
        #     motor_parameters = {'timestamps': self.watch_timestamps,
        #                         'velocity': self.velocities[motor_name],
        #                         'positions': self.watch_positions[motor_name]}
        #     motor_params_dict[motor_name] = motor_parameters
        #
        # data = {f'{self.name}_{self.detector.channel1.rois.roi01.name}': self.watch_intensities,
        #         f'{self.name}_parameters': motor_params_dict}
        #
        # now = ttime.time()
        # yield {'data': data,
        #        'timestamps': {key: now for key in data}, 'time': now,
        #        'filled': {key: False for key in data}}

    def _watch_function(self, *args, **kwargs):
        watch_pos, watch_int, watch_time = self.watch_func(self.motors, self.detector)
        for motor_name, field in self.motors.items():
            for field_name, val in field.items():
                self.watch_positions[motor_name][field_name].extend(watch_pos[motor_name][field_name])
        self.watch_intensities.extend(watch_int)
        self.watch_timestamps.extend(watch_time)

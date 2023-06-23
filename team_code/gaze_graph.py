# Lead vehicle
# Traffic lights/stop signs
# Vanishing point of road
# Oncoming vehicle
# Turning: (possibly ignore initially:)
# Intermediate waypoints (arc, use existing data)
import sys
import copy
import numpy as np

# compute transitions according to the following:
# IF there is a higher priority object: transition to higher priority with @param upper_transition_probability, spread uniformly over the objects
# ELSE IF the current object is still in the scene: self-transition with stay_probability, switch to a uniformly random other object
# ELSE IF the current object is no longer in the scene: transition to the highest priority object with upper_transition probability + uniform random, renormalized


class GazeTracker():
    def __init__(self, initial_fixate_probability, decay, upper_transition_probability):
        self.current_state = "vanishing_point"
        self.stay_probability = initial_fixate_probability
        self.initial_fixate_probability = initial_fixate_probability
        self.decay = decay
        self.upper_transition_probability = upper_transition_probability
        # TODO: add intermediate waypoints/turning
        self.class_order = ["lead_vehicle", "walker", "stop_sign", "traffic_light", "vanishing_point", "oncoming_vehicle"]
        self.min_distances = { # TODO: fill in these values with reasonable numbers
            "lead_vehicle": 30.0,
            "walker": 20.0,
            "traffic_light": 20.0,
            "stop_sign": 10.0,
            "oncoming_vehicle": 10.0
        }
        default_center = 0.23
        self.angle_range = {
            "lead_vehicle": [-default_center,default_center],
            "walker": [-1.5,1.5],
            "traffic_light": [-default_center,default_center],
            "stop_sign": [0.0,0.5],
            "oncoming_vehicle": [-default_center,default_center],
        }

    def get_class_name(self, obj_name): # turns an instance into a class
        return obj_name.strip("0123456789")

    def check_in(self, obj_name, distang):
        '''
        generates an existance vector
        '''
        dist, angle = distang
        class_name = self.get_class_name(obj_name)
        return (class_name, 
                obj_name, 
                ((dist < self.min_distances[class_name]) 
                and (self.angle_range[class_name][0] < angle < self.angle_range[class_name][1]))
                , dist)


    def update_graph(self, observed_values, action_waypoints):
        '''
        @param objects: a python dict with the name of the object mapped to the tuple of distance away and gaze angle
        @param action_waypoints: the set of waypoints the agent is taking, TODO: implement this
        '''
        existance_vector = [self.check_in(*obj_dist) for obj_dist in observed_values.items()]
        existance_names = {obj_dist[0]: self.check_in(*obj_dist)[2] for obj_dist in observed_values.items()}
        existance_names["vanishing_point"] = True
        no_stay = not (self.current_state in existance_names and existance_names[self.current_state]) or self.stay_probability <= 0
        existance_dict = {"vanishing_point": ("vanishing_point", 0)} # a dictionary of class_name -> (in class)
        for ev in existance_vector:
            if ev[2]: # the object is within minimal distance
                if not (ev[0] == self.get_class_name(self.current_state) and not no_stay): # the new object is not of the same class as the current object
                    if ev[0] in existance_dict: # an instance of the object already exists
                        if ev[3] < existance_dict[ev[0]][1]: # the new object is closer than the current one
                            existance_dict[ev[0]] = (ev[1], ev[3])
                        # otherwise don't add the value
                    else:
                        existance_dict[ev[0]] = (ev[1], ev[3])
                else:
                    existance_dict[ev[0]] = (ev[1], ev[3])
        existance_order = list(existance_dict.values())
        existance_order.sort(key = lambda x: x[-1])
        print("eo", existance_order, existance_dict)

        existance_vector = {cn: (cn in existance_dict) for cn in self.class_order}
        existance_vector["vanishing_point"] = True # vanishing point has no corresponding object

        # print(self.current_state, self.class_order)
        stay_index = self.class_order.index(self.get_class_name(self.current_state))
        num_seen = 0
        upper_transitions = list()
        lower_transitions = list()
        for i, c in enumerate(self.class_order):
            # if the class order object is there
            if existance_vector[c]:
                if i < stay_index: # there is a higher priority object
                    upper_transitions.append(existance_dict[c][0])
                elif i > stay_index:
                    lower_transitions.append(existance_dict[c][0])

        object_probabilities = dict()
        total_probability = 0
        if no_stay: # transition to a new state when there is no object to stay with
            first = True
            for c in self.class_order:
                if c in existance_dict:
                    if first:
                        object_probabilities[c] = self.upper_transition_probability
                        first = False
                    else:
                        object_probabilities[c] = 1
                    total_probability += object_probabilities[c]
            object_probabilities = {c: object_probabilities[c] / total_probability  for c in object_probabilities.keys()}
            self.stay_probability = self.initial_fixate_probability
        else: # there is an object to stay with
            new_stay_probability = self.stay_probability - self.decay
            
            # print(self.stay_probability, self.decay)

            if len(lower_transitions) > 0:
                for c in upper_transitions:
                    object_probabilities[c] = (1-new_stay_probability) * self.upper_transition_probability / len(upper_transitions)
            else:
                for c in upper_transitions:
                    object_probabilities[c] = (1-new_stay_probability) / len(upper_transitions)
            if len(upper_transitions) > 0:
                for c in lower_transitions:
                    object_probabilities[c] = (1-new_stay_probability) * (1-self.upper_transition_probability) / len(lower_transitions)
            else:
                for c in lower_transitions:
                    object_probabilities[c] = (1-new_stay_probability) / len(lower_transitions)
            if len(lower_transitions) == len(upper_transitions) == 0:
                new_stay_probability = 1
            object_probabilities[self.current_state] = new_stay_probability

        ops = [(c, p) for c,p in object_probabilities.items()]
        names = [op[0] for op in ops]
        ps = [op[1] for op in ops]
        # print(ops, ps, existance_dict, np.random.choice(names, p=ps), np.random.choice(names, p=ps) in existance_dict)
        new_current_state = existance_dict[self.get_class_name(np.random.choice(names, p=ps))][0] # the name of the object
        self.stay_probability -= self.decay
        if new_current_state != self.current_state:
            vanishing_penalty = 0.2 if new_current_state == "vanishing_point" else 0# reduces likelihood of staying on vanishing point
            self.stay_probability = self.initial_fixate_probability - vanishing_penalty
        self.current_state = new_current_state
        print(self.current_state, self.stay_probability)
    
    def return_state(self):
        return self.current_state

def type_hash(type_name):
    if type_name.find('vehicle') != -1:
        return 'lead_vehicle' # TODO: assumes all vehicles are lead vehicles for now
    elif type_name.find('walker') != -1:
        return 'walker'
    elif type_name.find('traffic_light') != -1:
        return 'traffic_light'
    elif type_name.find('stop') != -1:
        return 'stop_sign'
    else:
        return "n/a"

def read_file(file_name):
    with open(file_name, mode ='r')as f:
        datapoints = list()
        current_values = dict()
        current_step = 0
        for line in f:
            lstep, id, obj_type, distance, xyangle = line.split('/')
            
            # add the current step to the dataset
            if int(lstep) != current_step:
                datapoints.append(current_values)
                current_step = int(lstep)
                current_values = dict()
            
            obj_type = type_hash(obj_type)
            if obj_type != "n/a":
                # current_values[obj_type + id] = [float(distance), float(xyangle)] # id should be unique per time step
                current_values[obj_type + id] = float(distance) # id should be unique per time step
    
    # print(datapoints)
    return datapoints
            
if __name__ == "__main__":
    gaze_path = sys.argv[1]
    datapoints = read_file(gaze_path)
    tracker = GazeTracker(.99, 0.01, 0.8)
    for i in range(len(datapoints)):
        dp = datapoints[i]
        tracker.update_graph(dp, None)
        rs = tracker.return_state()
        print(rs)
        
        
        
        
        
        
        
        
        
        
        

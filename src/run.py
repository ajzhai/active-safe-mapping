#!/usr/bin/python
import ray
import rospy
import os
from ray import tune
from ray.tune import grid_search
from gym_randrooms import RandomRooms

MAP_SCALE = 100  # pixels per position unit
TIME_WEIGHT = 0.2
EPISODE_LENGTH = 120  # in seconds
IMG_INTERVAL = 0.5  # in position units
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

if __name__ == "__main__":
    rospy.init_node('reinforcement_learner', anonymous=True)
    rospy.sleep(10.)
    ray.init()
    tune.run(
        "PPO",
        resources_per_trial={
            "cpu": 32,
            "gpu": 0
        },
        stop={
            "timesteps_total": 2048
        },
        config={
            "env": RandomRooms,  # or "corridor" if registered above
            "sample_batch_size": 128,
            "train_batch_size": 128,
            "sgd_minibatch_size": 128,
            "lr": 1e-2,  # grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            "num_workers": 0,  # parallelism
            "num_cpus_per_worker": 2,
            "num_gpus": 1,
            "env_config": {
                "map_scale": MAP_SCALE,
                "time_weight": TIME_WEIGHT,
                "ep_length": EPISODE_LENGTH,
                "img_interval": IMG_INTERVAL
            },
        },
    )

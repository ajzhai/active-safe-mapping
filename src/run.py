#!/usr/bin/python
import ray
import rospy
from ray import tune
from ray.tune import grid_search
from gym_randrooms import RandomRooms

MAP_SCALE = 100  # pixels per position unit
TIME_WEIGHT = 0.2
EPISODE_LENGTH = 120  # in seconds
IMG_INTERVAL = 0.5  # in position units

if __name__ == "__main__":
    rospy.init_node('reinforcement_learner', anonymous=True)
    rospy.sleep(10.)
    ray.init(num_cpus=4, num_gpus=2)
    tune.run(
        "PPO",
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
            "num_cpus_per_worker": 4,
            "num_gpus": 0,
            "env_config": {
                "map_scale": MAP_SCALE,
                "time_weight": TIME_WEIGHT,
                "ep_length": EPISODE_LENGTH,
                "img_interval": IMG_INTERVAL
            },
        },
    )

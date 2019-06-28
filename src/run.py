import ray
import rospy
from ray import tune
from ray.tune import grid_search
from gym_randrooms import RandomRooms

MAP_SCALE = 100  # pixels per position unit
TIME_WEIGHT = 0.2
EPISODE_LENGTH = 100  # in seconds
IMG_INTERVAL = 0.2  # in position units

if __name__ == "__main__":
    rospy.init_node('reinforcement_learner', anonymous=True)
    ray.init()
    tune.run(
        "PPO",
        stop={
            "timesteps_total": 10000,
        },
        config={
            "env": RandomRooms,  # or "corridor" if registered above
            "lr": 1e-4,  # grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            "num_workers": 1,  # parallelism
            "env_config": {
                "map_scale": MAP_SCALE,
                "time_weight": TIME_WEIGHT,
                "ep_length": EPISODE_LENGTH,
                "img_interval": IMG_INTERVAL
            },
        },
    )

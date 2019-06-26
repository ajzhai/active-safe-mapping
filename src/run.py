import ray
from ray import tune
from ray.tune import grid_search
from gym_randrooms import RandomRooms

ROOMS_DIR = ''
IMG_SCALE = 100
EPISODE_LENGTH = 100

if __name__ == "__main__":
    ray.init()
    tune.run(
        "PPO",
        stop={
            "timesteps_total": 10000,
        },
        config={
            "env": RandomRooms,  # or "corridor" if registered above
            "lr": grid_search([1e-2, 1e-4, 1e-6]),  # try different lrs
            "num_workers": 1,  # parallelism
            "env_config": {
                "rooms_dir": ROOMS_DIR,
                "img_scale": IMG_SCALE,
                "ep_length": EPISODE_LENGTH
            },
        },
    )

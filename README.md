# active-safe-mapping
ROS package for simulating UAV landing area exploration with visual sensing. Implements novel reinforcement-based active learning paradigm. Requires ROS, Gym, Ray.

The safe mapping problem for UAV landing is defined as follows: given an unknown area, create an (x, y)-map of where it is safe to land and
where it is not. Our goal is to use visual information to both create this model and plan the flight trajectory. In our main approach, 
the underlying safe-mapper model is a LSTM-CNN which learns from labeled data collected by sampling from the environment according
to the decisions of an RL agent which gets rewards from the improvement of the safe-mapper model. This package includes the algorithm
implementation and tools for testing, especially in Gazebo simulation.

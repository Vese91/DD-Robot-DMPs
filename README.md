## Description
Code for replicating the experiments of the paper (please cite):
```bibtex
@article{vesentini2025dynamic,
  title={Dynamic Movement Primitives with Control Barrier Functions for Constrained Trajectory Planning},
  author={Vesentini, Federico and Meli, Daniele and Sansonetto, Nicola and Di Persio, Luca and Muradore, Riccardo},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE},
  note={In publication}
}
```

## Requirements
- Python 3.11 tested
- Clone and install the original DMP repo from https://gitlab.com/altairLab/new_pydmps.git

## Run
- Plot generation for obstacle avoidance with moving obstacles
    ```bash 
    python test_obstacle_sim.py
    ```
- Plot generation for the constraints on maximum Cartesian velocity and centrifugal acceleration (to avoid slipping)
    ```bash 
    python test_centrifugal_and_max_vel.py
    ```
- Plot generation for real robot obstacle avoidance (accompanying video of the paper)
    ```bash 
    python test_real_robot.py
    ```

## Contacts
federico.vesentini@univr.it \
daniele.meli@univr.it
# Robocop
by Daniel Horcajo
## _Brace yourselves - autonomous robots are here_

Robocop is an open-source library that allows a robot find its way out of a maze autonomously - all while vacuuming and mopping the floor!

## Features

- Import you own custom map of the house
- Make the robot dance and beep along to any song of your choice
- Find Waldo
- Control linear and angular velocities

## Execution

You know what to do.

```sh
cd src/
python main.py
```

## Development history

This project is a fork of the Cyber Physical Systems & Robotics course from Comillas University. 

Steps: 
1. Start off using the original files of the CPSR project library
2. Add custom methods to the Map and Robot classes, such as `map.find_waldo()`
3. Create main execution file to run and launch the robot device
4. Add map data
5. Add dancing capabilities to the robot by creating the `feat_robot` branch. **DISCLAIMER**: one's self-steem may be damaged after seeing this bad boy's moves.
6. Modify the `Robot.move()` method to simulate actual displacement on the map. 
7. Resolve merge conflicts from `development` into `release` because _someone_ modified the `Robot.move()` method _by mistake_ in the `release` branch... Interns.
8. Perform various test in the sandbox file provided and record whether they were successful (`OK`) or unsuccessful (`ERROR`).
9. Bring changes to `main`, add requirements and deploy into production.
10. `Done!`

### Git commit history tree view:

![Git commit history](https://github.com/dannyhorcajo/datahack_prod_alg/blob/main/src/img/git_history.png?raw=true)

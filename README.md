# srl_robosuite
Extension To OG Robosuite with GT Safe Robotics Lab Robots.

# Pixi Setup Guide

This project supports [pixi](https://pixi.sh/) for reproducible environment management, replacing `uv` and `pip` + `conda` combinations.

## 1. Prerequisites

If you don't have pixi installed, install it via:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

## 2. Installation

Run the following command at the root of the workspace. This will solve the environment and install all dependencies (both Conda and PyPI) into a local `.pixi` directory.

```bash
pixi install
```

## 3. Usage

To run scripts using the environment, prefix commands with `pixi run`.

**Running a specific python script:**
```bash
pixi run python projects/experiment_envs/exp_dual_kinova3_oscmink_keyboard.py
```

**Opening a shell inside the environment:**
```bash
pixi shell
# Now you can run `python ...` directly
```

**Using VS Code:**
1. Install the "Pixi" extension for VS Code.
2. It should automatically detect the environment (or you can select the python interpreter inside .pixi/envs/default/bin/python).
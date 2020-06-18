"""Run HalfCheetah MPC."""

import os

from lsf_runner import init_runner, make_commands

from exps.gpucrl.half_cheetah import ACTION_COST

runner = init_runner(
    f"GPUCRL_HalfCheetah_mpc", num_threads=1, wall_time=1439, num_workers=12
)

cmd_list = make_commands(
    "mpc.py",
    base_args={"num-threads": 1},
    fixed_hyper_args={},
    common_hyper_args={
        "seed": [142],
        "exploration": ["thompson", "optimistic", "expected"],
        "model-kind": ["ProbabilisticEnsemble"],  # Deterministic Ensemble
        "action-cost": [0, ACTION_COST, 5 * ACTION_COST, 10 * ACTION_COST],
    },
    algorithm_hyper_args={},
)
runner.run(cmd_list)
if "AWS" in os.environ:
    os.system("sudo shutdown")

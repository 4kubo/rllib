from dotmap import DotMap

from exps.gpucrl.inverted_pendulum import TRAIN_EPISODES, ENVIRONMENT_MAX_STEPS, ACTION_COST, \
    get_agent_and_environment
from exps.gpucrl.mb_mppo_arguments import parser
from exps.gpucrl.inverted_pendulum.plotters import plot_pendulum_trajectories
from exps.gpucrl.util import train_and_evaluate
from exps.gpucrl.plotters import set_figure_params

PLAN_HORIZON, SIM_TRAJECTORIES = 1, 8

parser.description = 'Run Swing-up Inverted Pendulum using Model-Based MPPO.'
parser.set_defaults(action_cost=ACTION_COST,
                    train_episodes=TRAIN_EPISODES,
                    environment_max_steps=ENVIRONMENT_MAX_STEPS,
                    plan_horizon=PLAN_HORIZON,
                    sim_num_steps=ENVIRONMENT_MAX_STEPS,
                    sim_initial_states_num_trajectories=SIM_TRAJECTORIES // 2,
                    sim_initial_dist_num_trajectories=SIM_TRAJECTORIES // 2,
                    model_kind='ProbabilisticEnsemble',
                    model_learn_num_iter=50,
                    model_opt_lr=1e-3,
                    seed=0)

args = parser.parse_args()
params = DotMap(vars(args))

environment, agent = get_agent_and_environment(params, 'mbmppo')
set_figure_params(serif=True, fontsize=9)
train_and_evaluate(agent, environment, params)
                   # save_milestones=list(range(params.train_episodes)),
                   # plot_callbacks=[plot_pendulum_trajectories])

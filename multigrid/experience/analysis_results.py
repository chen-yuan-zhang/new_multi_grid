import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    # datasets = ["3scenarios_small.csv", "5scenarios_small.csv", "7scenarios_small.csv", "3scenarios_medium.csv", "5scenarios_medium.csv", "10scenarios_medium.csv"]
    # algorithms = ["random", "coverage", "greedy", "agrmcts_goal_max"]
    target_limit = 450
    datasets = [f"pure_rl_evaluation_results_temp_{target_limit}_formal_dataset_v0.csv", f"/home/sukai/Project/chenyuan_project/new_multi_grid/evaluation_results_temp_{target_limit}_formal_dataset_v0.csv", f'ns_only_in_view_evaluation_results_temp_{target_limit}_formal_dataset_v0.csv', f'symbolic_evaluation_results_temp_{target_limit}_formal_dataset_v0.csv']
    
    # remove if file not exist
    import os
    datasets = [d for d in datasets if os.path.exists(d)]
    print(f"Found datasets size: {len(datasets)}")

    for dataset in datasets:
        all_algorithm_scnerios = []
        final_prob_results = []
        convergence_results = []
        success_rate_results = []

        # passive_final_prob_results = []
        # passive_convergence_results = []
        # passive_success_rate_results = []

        #print(f"Dataset: {dataset}")
        scenarios = pd.read_csv(dataset)
        # limit scenario to first rows 
        scenarios = scenarios.head(target_limit)
        vals = []
        succs = []
        for j in range(len(scenarios)):
            vals.append(scenarios.loc[j, "eval_convergence_step"])
            succs.append(scenarios.loc[j, "eval_success"])
        convergence = np.mean(vals)
        success_rate = np.mean(succs)
        avg_exec_times = scenarios["eval_execution_time"].mean()
        avg_visibility_ratio = scenarios["visibility_ratio"].mean()
        avg_final_confidence_value = scenarios["final_confidence"].mean()

        print(f"Dataset: {dataset}, Results Analysis using {len(scenarios)}/720 instances.\nConvergence: {convergence:.2f}\nSuccess: {success_rate*100:.2f}%\nAvg Exec Time: {avg_exec_times:.2f}s\nAvg Visibility Ratio: {avg_visibility_ratio:.2f}\nAvg Final Confidence: {avg_final_confidence_value:.2f}")
        
        # Read scenarios for each algorithm
        # for algorithm in algorithms:
        #     scenarios = pd.read_csv("results_" + algorithm + "_" + dataset.split('.')[0] + "_acc_True.csv")
        #     all_algorithm_scnerios.append(scenarios)

        # # Calculate final probabilities, convergence, and success rates for each algorithm
        # for i, algorithm in enumerate(algorithms):
        #     scenarios = all_algorithm_scnerios[i]
        #     final_prob = scenarios["final_prob"].mean()

        #     # calculate convergence
        #     vals = []
        #     for j in range(len(scenarios)):
        #         if scenarios.loc[j, "first_step"] > 0:
        #             vals.append(1 - scenarios.loc[j, "first_step"]/ scenarios.loc[j, "total_step"])
        #         else:
        #             vals.append(0)

        #     convergence = np.mean(vals)

        #     succs = []
        #     for j in range(len(scenarios)):
        #         if scenarios.loc[j, "first_step"] > 0:
        #             succs.append(1)
        #         else:
        #             succs.append(0)
        #     success_rate = np.mean(succs)

            # # calculate passive final probabilities, convergence, and success rates
            # passive_final_prob = scenarios["passive_final_prob"].mean()
            # passive_vals = []
            # for j in range(len(scenarios)):
            #     if scenarios.loc[j, "passive_first_step"] > 0:
            #         passive_vals.append(1 - scenarios.loc[j, "passive_first_step"]/ scenarios.loc[j, "total_step"])
            #     else:
            #         passive_vals.append(0)
            # passive_convergence = np.mean(passive_vals)
            # passive_succs = []
            # for j in range(len(scenarios)):
            #     if scenarios.loc[j, "passive_first_step"] > 0:
            #         passive_succs.append(1)
            #     else:
            #         passive_succs.append(0)
            # passive_success_rate = np.mean(passive_succs)

            # final_prob_results.append(final_prob)
            # convergence_results.append(convergence)
            # success_rate_results.append(success_rate)

            # passive_final_prob_results.append(passive_final_prob)
            # passive_convergence_results.append(passive_convergence)
            # passive_success_rate_results.append(passive_success_rate)

            # only print convergence
            # print(f"Algorithm: {algorithm}, Convergence: {convergence:.2f}")
            # print(f"Passive Algorithm: {algorithm}, Passive Convergence: {passive_convergence:.2f}")





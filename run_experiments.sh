#!/bin/bash
set -x
# for policy in 1 2 3 4
for policy in 6
do
	extra_args=""
	echo $policy
	if [[ $policy = 1 ]]
	then
		project_name=exp0090_politica_1_oracle_1m1l
	elif [[ $policy = 2 ]]
	then
		project_name=exp0090_politica_2_holdout_1m1l
	elif [[ $policy = 3 ]]
	then
		project_name=exp0090_politica_3_best_validation_1m1l
	elif [[ $policy = 4 ]]
	then
		project_name=exp0090_politica_4_diff_best_holdout_1m1l
	elif [[ $policy = 5 ]]
	then
		project_name=exp0090_politica_5_oracle_1m5l_mon_metric_test_topsis_pareto_append_orig_inp
		extra_args="model.max_layers=5 model.stack_hidden_layers=True training.monitored_metric_add_layers=test_overall_acc model.transform_data_strategy=append_original_input"
	elif [[ $policy = 6 ]]
	then
		project_name=exp0090_politica_6_topsis_pareto_holdout_1m1l
	elif [[ $policy = 7 ]]
	then
		project_name=exp0090_politica_7_topsis_pareto_holdout_with_mean_diff_1m1l
	fi


WANDB_MODE=online python parallel_mlps/main.py --multirun training.data_home=/home/fcf/projects/parallel_mlps/datasets/ training.dataset="Australian,balance-scale,blood-transfusion-service-center,car\(3\),climate-model-simulation-crashes\(4\),credit-g,diabetes,hill-valley,ilpd,ionosphere,libras_move,lsvt,musk,satimage,wdbc" training.experiment_num=range\(0,20\) model.topk=1 training.debug_test=True training.pareto_frontier=False model.max_layers=1 training.distance_name=correlation model.chosen_policy=policy${policy} training.project_name=${project_name} $extra_args

WANDB_MODE=online python parallel_mlps/main.py --multirun training.data_home=/home/fcf/projects/parallel_mlps/datasets/ training.dataset="Australian,balance-scale,blood-transfusion-service-center,car\(3\),climate-model-simulation-crashes\(4\),credit-g,diabetes,hill-valley,ilpd,ionosphere,libras_move,lsvt,musk,satimage,wdbc" training.experiment_num=range\(0,20\) model.topk=1 training.debug_test=True training.pareto_frontier=False model.max_layers=1 training.distance_name=null model.chosen_policy=policy${policy} training.project_name=${project_name}_nosbss $extra_args
done


#WANDB_MODE=online python parallel_mlps/main.py --multirun training.data_home=/home/fcf/projects/parallel_mlps/datasets/ training.dataset="Australian,balance-scale,blood-transfusion-service-center,car\(3\),climate-model-simulation-crashes\(4\),credit-g,diabetes,hill-valley,ilpd,ionosphere,libras_move,lsvt,musk,satimage,wdbc,vowel\(2\)" training.experiment_num=range\(0,20\) model.topk=1 training.debug_test=True training.pareto_frontier=False model.max_layers=1 training.distance_name=correlation model.chosen_policy=policy1 training.project_name=exp0090_politica_1_oracle_1m1l
#WANDB_MODE=online python parallel_mlps/main.py --multirun training.data_home=/home/fcf/projects/parallel_mlps/datasets/ training.dataset="Australian,balance-scale,blood-transfusion-service-center,car\(3\),climate-model-simulation-crashes\(4\),credit-g,diabetes,hill-valley,ilpd,ionosphere,libras_move,lsvt,musk,satimage,wdbc,vowel\(2\)" training.experiment_num=range\(0,20\) model.topk=1 training.debug_test=True training.pareto_frontier=False model.max_layers=1 training.distance_name=null model.chosen_policy=policy3 training.project_name=exp0090_politica_3_best_validation_1m1l_nosbss
#WANDB_MODE=online python parallel_mlps/main.py --multirun training.data_home=/home/fcf/projects/parallel_mlps/datasets/ training.dataset="Australian,balance-scale,blood-transfusion-service-center,car\(3\),climate-model-simulation-crashes\(4\),credit-g,diabetes,hill-valley,ilpd,ionosphere,libras_move,lsvt,musk,satimage,wdbc,vowel\(2\)" training.experiment_num=range\(0,20\) model.topk=1 training.debug_test=True training.pareto_frontier=False model.max_layers=1 training.distance_name=correlation model.chosen_policy=policy4 training.project_name=exp0090_politica_4_best_diff_holdout_1m1l
#WANDB_MODE=online python parallel_mlps/main.py --multirun training.data_home=/home/fcf/projects/parallel_mlps/datasets/ training.dataset="Australian,balance-scale,blood-transfusion-service-center,car\(3\),climate-model-simulation-crashes\(4\),credit-g,diabetes,hill-valley,ilpd,ionosphere,libras_move,lsvt,musk,satimage,wdbc,vowel\(2\)" training.experiment_num=range\(0,20\) model.topk=1 training.debug_test=True training.pareto_frontier=False model.max_layers=1 training.distance_name=null model.chosen_policy=policy4 training.project_name=exp0090_politica_4_best_diff_holdout_1m1l_nosbss


#WANDB_MODE=online python parallel_mlps/main.py --multirun training.data_home=/home/fcf/projects/parallel_mlps/datasets/ training.dataset="Australian,balance-scale,blood-transfusion-service-center,car\(3\),climate-model-simulation-crashes\(4\),credit-g,diabetes,hill-valley,ilpd,ionosphere,libras_move,lsvt,musk,satimage,wdbc,vowel\(2\)" training.experiment_num=range\(0,20\) model.topk=1 training.debug_test=True training.pareto_frontier=False model.max_layers=1 training.distance_name=correlation model.chosen_policy=policy3 training.project_name=exp0090_politica_3_best_validation_1m1l
#WANDB_MODE=online python parallel_mlps/main.py --multirun training.data_home=/home/fcf/projects/parallel_mlps/datasets/ training.dataset="Australian,balance-scale,blood-transfusion-service-center,car\(3\),climate-model-simulation-crashes\(4\),credit-g,diabetes,hill-valley,ilpd,ionosphere,libras_move,lsvt,musk,satimage,wdbc,vowel\(2\)" training.experiment_num=range\(0,20\) model.topk=1 training.debug_test=True training.pareto_frontier=False model.max_layers=1 training.distance_name=null model.chosen_policy=policy3 training.project_name=exp0090_politica_3_best_validation_1m1l_nosbss
#WANDB_MODE=online python parallel_mlps/main.py --multirun training.data_home=/home/fcf/projects/parallel_mlps/datasets/ training.dataset="Australian,balance-scale,blood-transfusion-service-center,car\(3\),climate-model-simulation-crashes\(4\),credit-g,diabetes,hill-valley,ilpd,ionosphere,libras_move,lsvt,musk,satimage,wdbc,vowel\(2\)" training.experiment_num=range\(0,20\) model.topk=1 training.debug_test=True training.pareto_frontier=False model.max_layers=1 training.distance_name=correlation model.chosen_policy=policy4 training.project_name=exp0090_politica_4_best_diff_holdout_1m1l
#WANDB_MODE=online python parallel_mlps/main.py --multirun training.data_home=/home/fcf/projects/parallel_mlps/datasets/ training.dataset="Australian,balance-scale,blood-transfusion-service-center,car\(3\),climate-model-simulation-crashes\(4\),credit-g,diabetes,hill-valley,ilpd,ionosphere,libras_move,lsvt,musk,satimage,wdbc,vowel\(2\)" training.experiment_num=range\(0,20\) model.topk=1 training.debug_test=True training.pareto_frontier=False model.max_layers=1 training.distance_name=null model.chosen_policy=policy4 training.project_name=exp0090_politica_4_best_diff_holdout_1m1l_nosbss



#WANDB_MODE=online python parallel_mlps/main.py --multirun training.data_home=/home/fcf/projects/parallel_mlps/datasets/ training.dataset="Australian,balance-scale,blood-transfusion-service-center,car\(3\),climate-model-simulation-crashes\(4\),credit-g,diabetes,hill-valley,ilpd,ionosphere,libras_move,lsvt,musk,satimage,wdbc,vowel\(2\)" training.experiment_num=range\(0,20\) model.topk=1 training.debug_test=True training.pareto_frontier=False model.max_layers=1 training.distance_name=null training.project_name=exp0088_politica_2_best_holdout_1m1l_nosbss

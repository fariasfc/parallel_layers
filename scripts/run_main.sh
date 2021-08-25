if [ $# -lt 1 ] ; then
    echo 'Need to pass project_name'
    exit 1
fi

cd /home/fcf/projects/representation_learner

. ./.env
# echo $WANDB_API_KEY
# echo $WANDB_MODE
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python src/main.py --multirun ${EXTRA_ARGS} training.project_name=$1
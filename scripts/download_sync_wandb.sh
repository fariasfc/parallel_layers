if [[ $# -eq 0 ]] ; then
    echo 'Need to pass project_name'
    exit 1
fi

echo $1

. ../.env

ssh fcf@200.133.5.120 "cd /home/fcf/projects/parallel_mlps/outputs; ../scripts/tar_wandb.sh ${1}"
# Downloading wandb runs
rsync -avH -e ssh fcf@200.133.5.120:/home/fcf/projects/parallel_mlps/outputs/tar_wandb_runs/$1.tar /tmp/$1.tar

# extracting file
cd /tmp
tar -xvf $1.tar
cd /tmp/outputs

# Syncing runs
find /tmp/${1} -type d -regex ".*\/${1}\/.*offline-run-[0-9]\{8\}_[0-9]\{6\}-.\{8\}" -exec wandb sync {} \;



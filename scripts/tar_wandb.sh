if [[ $# -eq 0 ]] ; then
    echo 'Need to pass project_name'
    exit 1
fi

echo "tar_wandb.sh running"
echo $1

mkdir -p tar_wandb_runs

#find . -type d -regextype sed -regex ".*\/${1}\/.*offline-run-[0-9]\{8\}_[0-9]\{6\}-.\{8\}" -exec tar -C . -rvf tar_wandb_runs/$1.tar {} \;
files=$(find . -type f -regextype sed -regex ".*\/${1}\/.*offline-run-[0-9]\{8\}_[0-9]\{6\}-.\{8\}.*.wandb$")
# | xargs -n 1 dirname)

#echo $files
for file in $files; do
	dir=$(dirname $file)
	echo "${dir} saving"
	tar rvf tar_wandb_runs/$1.tar $dir
done

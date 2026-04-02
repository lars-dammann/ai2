cd "$(dirname "$0")"

task=$1
comment=$2

# Pull the latest simulation from git
git pull

datetime=$(date +%Y%m%d-%H%M%S)

# Move simulation files to the designated folder
mkdir $PWD/transfer/$datetime
cp -r ai2 $PWD/transfer/$datetime/.
cp -r configs $PWD/transfer/$datetime/.

# Add information to an info.txt file
echo -e "Comment: $comment\n" >> $PWD/transfer/$datetime/info.txt
echo -e "Git info:" >> $PWD/transfer/$datetime/info.txt
echo -e $(git config --get remote.origin.url) >> $PWD/transfer/$datetime/info.txt
echo -e "Git branch:" >> $PWD/transfer/$datetime/info.txt
echo -e $(git rev-parse --abbrev-ref HEAD) >> $PWD/transfer/$datetime/info.txt
echo -e "Git commit:" >> $PWD/transfer/$datetime/info.txt
echo -e $(git show --oneline -s) >> $PWD/transfer/$datetime/info.txt

# Run the specified task with the right current folder
sbatch slurm/$task.sbatch $datetime $task
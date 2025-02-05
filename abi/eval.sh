#! /bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=1-6

case $SLURM_ARRAY_TASK_ID in
   1)  ARGS='--model=m1a --checkpoint_prefix=affine_lowN' ;;
   2)  ARGS='--model=m2  --checkpoint_prefix=affine_lowN' ;;
   3)  ARGS='--model=m3  --checkpoint_prefix=affine_lowN' ;;
   4)  ARGS='--model=m4b --checkpoint_prefix=affine_lowN' ;;
   5)  ARGS='--model=m5  --checkpoint_prefix=affine_lowN' ;;
   6)  ARGS='--model=m6  --checkpoint_prefix=affine_lowN' ;;
esac


module load mamba

source activate gin
python eval.py $ARGS
conda deactivate

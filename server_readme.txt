to run the code on server run

$sbatch --ntasks=8  --time=12:00:00  --mem-per-cpu=1024  --output="/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/output_file"  --error="/cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/logs/error_file"  --open-mode=truncate --wrap="python3 /cluster/home/mnobar/code/spinningup/spinup/examples/pytorch/Hybrid_SAC.py PYTHONUNBUFFERED=1"

# Ginsburg SLURM cheatsheet (p2-walrus)

Account: `astro`. UNI: `sod2112`. Workdir: `/burg/astro/users/sod2112/p2-walrus/`.

## Submit

```bash
sbatch slurm/walrus_zero_shot.sh
# Returns: "Submitted batch job 12345678"
```

## Monitor queue

```bash
# All my jobs
squeue -u sod2112

# Pending jobs with estimated start time
squeue -u sod2112 --start

# Single job, full info (incl. ReasonStart)
scontrol show job <jobid>

# Partition state (how many idle/alloc/down)
sinfo -s

# How busy GPU partitions are
sinfo -p gpu -o "%P %D %t %G"
```

## During run

```bash
# Tail the SLURM output log
tail -f slurm/walrus_zero_shot_<jobid>.out

# Check resource utilization (only while running)
ssh ginsburg "ssh <nodename> nvidia-smi"   # nodename from squeue
sstat -j <jobid> --format=JobID,MaxRSS,MaxVMSize,AveCPU
```

## Post-mortem

```bash
# Historical accounting (works after job completes)
sacct -j <jobid> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS,ReqMem

# Why did it fail?
sacct -j <jobid> --format=JobID,State,Reason,Comment
```

## Cancel

```bash
scancel <jobid>
scancel -u sod2112             # all my jobs
scancel -u sod2112 -t pending  # only pending
```

## Interactive fallback (if queue >10 min)

```bash
# Request 1 A40 for 4 hours interactively
srun --pty -t 0-04:00 -A astro --gres=gpu:a40:1 --mem=64G -c 8 /bin/bash

# Same with A100
srun --pty -t 0-04:00 -A astro --gres=gpu:a100:1 --mem=64G -c 8 /bin/bash
```

## Cloud fallback (if Ginsburg blocked)

vast.ai with the same env. ~$0.40-0.80/hr A100. Use rsync to sync data + checkpoints.

```bash
# Local
ssh vast-p1 "mkdir -p /workspace/p2-walrus"
rsync -av --exclude env --exclude outputs ./ vast-p1:/workspace/p2-walrus/
```

## Decision tree

1. `sbatch` → pending >10 min via `squeue --start`?
2. Yes → try `srun --pty` interactive A40 → blocks >10 min?
3. Yes → try A100 partition → blocks >10 min?
4. Yes → cloud (vast.ai A100). Inference is 10s of minutes; cost is trivial.

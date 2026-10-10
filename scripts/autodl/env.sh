# Sourced by the instance-side scripts. torch sizes its CPU thread pool by the 176 cores the host
# shows, not by the instance's CPU quota: at the no-GPU 0.5 CPU, 91 threads made one update of a 2M
# model take three minutes.
export PATH=/root/miniconda3/bin:$PATH HF_HUB_OFFLINE=1
quota=$(awk '$1 != "max" {q = int($1 / $2); print (q < 1) ? 1 : q}' /sys/fs/cgroup/cpu.max 2>/dev/null || true)
if [ -n "$quota" ]; then export OMP_NUM_THREADS=$quota; fi

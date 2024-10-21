# tmp

## TODO

- [ ] test run (8B)
- [ ] fused kernel patch
- [ ] mup
- [ ] logger
    - [ ] activation / grad norm / param norm 
- [ ] dmoe
- [ ] zero1 shampoo


## example run

```bash
# create new venv
VENV_DIR=/path/to/dir/venv &&\
# VENV_DIR=/mnt/chatbot30TB/shseo/venv &&\
VENV_NAME=lingua &&\
python -m pip install --upgrade pip &&\
pip install virtualenv &&\
python -m virtualenv -p python3 $VENV_DIR/$VENV_NAME
```

```bash
VENV_DIR=/path/to/dir/venv &&\
VENV_NAME=lingua &&\
source $VENV_DIR/$VENV_NAME/bin/activate
```

```bash
# cd /path/to/dir/lingua
pip install -r requirements.txt
```

```bash
# cd /path/to/dir/lingua
# python ./setup/download_prepare_hf_data.py fineweb_edu_10bt
python ./setup/download_prepare_hf_data.py dclm_baseline_1.0
```

```bash
pip install wandb
wandb login
```

```bash
export LOCAL_RANK=0 &&\
export WORLD_SIZE=2 &&\
export MASTER_ADDR=node0 &&\
export MASTER_PORT=23458

export CONFIG=llama_1B
export DUMP_DIR="exp_logs/assets/logs/test_${CONFIG}_world_${WORLD_SIZE}"
torchrun --nproc-per-node $WORLD_SIZE \
-m apps.main.train \
config=apps/main/configs/${CONFIG}.yaml
```

```python
node0:23440:23664 [0] NCCL INFO threadThresholds 8/8/64 | 16/8/64 | 512 | 512
node0:23440:23664 [0] NCCL INFO 24 coll channels, 24 collnet channels, 0 nvls channels, 32 p2p channels, 32 p2p channels per peer
node0:23441:23666 [1] NCCL INFO NCCL_WORK_FIFO_DEPTH set by environment to 4194304.
node0:23440:23664 [0] NCCL INFO NCCL_WORK_FIFO_DEPTH set by environment to 4194304.
node0:23441:23666 [1] NCCL INFO TUNER/Plugin: Failed to find ncclTunerPlugin_v2, using internal tuner instead.
node0:23440:23664 [0] NCCL INFO TUNER/Plugin: Failed to find ncclTunerPlugin_v2, using internal tuner instead.
node0:23441:23666 [1] NCCL INFO ncclCommInitRank comm 0x55d761db2df0 rank 1 nranks 2 cudaDev 1 nvmlDev 1 busId 16000 commId 0xb3609cbb6ba7a88a - Init COMPLETE
node0:23440:23664 [0] NCCL INFO ncclCommInitRank comm 0x563d5b2b3360 rank 0 nranks 2 cudaDev 0 nvmlDev 0 busId 10000 commId 0xb3609cbb6ba7a88a - Init COMPLETE
1: INFO    24-10-19 18:46:56.490658 - 0:00:07 - Model size: 1,809,946,624 total parameters
0: INFO    24-10-19 18:46:56.491630 - 0:00:07 - Model size: 1,809,946,624 total parameters
1: INFO    24-10-19 18:46:56.492031 - 0:00:07 - GPU capacity: NVIDIA A100-SXM4-80GB (1) with 79.33GiB memory
0: INFO    24-10-19 18:46:56.492830 - 0:00:07 - GPU capacity: NVIDIA A100-SXM4-80GB (0) with 79.33GiB memory
1: INFO    24-10-19 18:46:56.494848 - 0:00:07 - GPU memory usage: NVIDIA A100-SXM4-80GB (1): 79.32501220703125 GiB capacity, 4.224609375 GiB peak, 5.325696470079506% peak
1: INFO    24-10-19 18:46:56.494932 - 0:00:07 - Starting build of optimizer...
1: INFO    24-10-19 18:46:56.496116 - 0:00:07 - Done with build of optimizer.
0: INFO    24-10-19 18:46:56.496769 - 0:00:07 - GPU memory usage: NVIDIA A100-SXM4-80GB (0): 79.32501220703125 GiB capacity, 4.224609375 GiB peak, 5.325696470079506% peak
0: INFO    24-10-19 18:46:56.496857 - 0:00:07 - Starting build of optimizer...
0: INFO    24-10-19 18:46:56.497909 - 0:00:07 - Done with build of optimizer.
0: INFO    24-10-19 18:46:56.508740 - 0:00:07 - Async dataloader started
0: INFO    24-10-19 18:46:56.509566 - 0:00:07 - Profiling active.  Traces will be saved at exp_logs/assets/logs/test_llama_1B_world_2/profiling
1: INFO    24-10-19 18:46:56.509680 - 0:00:07 - Async dataloader started
0: INFO    24-10-19 18:46:56.510721 - 0:00:07 - Starting MemSnapshotsProfilerWandb profiler...
1: INFO    24-10-19 18:46:56.510869 - 0:00:07 - Profiling active.  Traces will be saved at exp_logs/assets/logs/
1: INFO    24-10-19 18:46:56.509680 - 0:00:07 - Async dataloader started
0: INFO    24-10-19 18:46:56.510721 - 0:00:07 - Starting MemSnapshotsProfilerWandb profiler...
1: INFO    24-10-19 18:46:56.510869 - 0:00:07 - Profiling active.  Traces will be saved at exp_logs/assets/logs/test_llama_1B_world_2/profiling
1: INFO    24-10-19 18:46:56.511857 - 0:00:07 - Starting MemSnapshotsProfilerWandb profiler...
1: INFO    24-10-19 18:47:29.439328 - 0:00:40 - garbage collection
0: INFO    24-10-19 18:47:29.632364 - 0:00:40 - garbage collection
1: INFO    24-10-19 18:47:38.013589 - 0:00:49 - step: 1  acc: 0  loss: 12.2048  grad: 1.06e+01  flops: 4.25e+12  wps: 3.95e+02  iter:  8.4191  data: 32.8857  lr: 0.00e+00  mem: 51%  pow: 92.494 W
0: INFO    24-10-19 18:47:38.013923 - 0:00:49 - step: 1  acc: 0  loss: 12.2087  grad: 1.06e+01  flops: 4.25e+12  wps: 3.95e+02  iter:  8.2273  data: 33.1016  lr: 0.00e+00  mem: 51%  pow: 96.192 W
1: INFO    24-10-19 18:47:38.910772 - 0:00:50 - step: 2  acc: 0  loss: 12.1832  grad: 1.08e+01  flops: 1.97e+14  wps: 1.84e+04  iter:  0.8908  data: 0.001  lr: 6.00e-07  mem: 59%  pow: 392.956 W
0: INFO    24-10-19 18:47:38.911112 - 0:00:50 - step: 2  acc: 0  loss: 12.2032  grad: 1.08e+01  flops: 1.97e+14  wps: 1.83e+04  iter:  0.8907  data: 0.0015  lr: 6.00e-07  mem: 59%  pow: 398.257 W
1: INFO    24-10-19 18:47:39.823351 - 0:00:51 - step: 3  acc: 0  loss: 12.1364  grad: 1.00e+01  flops: 1.96e+14  wps: 1.82e+04  iter:  0.8975  data: 0.0009  lr: 1.20e-06  mem: 59%  pow: 393.815 W
0: INFO    24-10-19 18:47:39.823625 - 0:00:51 - step: 3  acc: 0  loss: 12.1798  grad: 1.00e+01  flops: 1.96e+14  wps: 1.82e+04  iter:   0.898  data: 0.0014  lr: 1.20e-06  mem: 59%  pow: 391.807 W
1: INFO    24-10-19 18:47:40.724381 - 0:00:51 - Shutting down MemSnapshotsProfilerWandb profiler...

...
```


## TBD
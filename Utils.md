# 2 Utilties

## 2.1 Logger

### 2.1.1 Logger Class
```python
class spinup.utils.logx.Logger(output_dir=None, output_fname='progress.txt', exp_name=None)
```
一个通用的日志记录器

```python
def __init__(output_dir=None, output_fname='progress.txt', exp_name=None)
```
output_dir (string):目录地址，默认为/tmp/experiments/some random number

output_fname (string)：一个以制表符为分割的文件，包含训练过程中记录的度量

exp_name (string): 实验名

```python
def save_config(self, config):
```

记录实验配置

```python
def log(self, msg, color='green'):
```

打印有色彩的消息

```python
def setup_tf_saver(self, sess, inputs, outputs):
```

```python
def save_state(self, state_dict, itr=None):
```



### 2.1.2 EpochLogger Class

```python
class EpochLogger(Logger):
```

记录每个epoch的average / std / min / max

```python
def store(self, **kwargs):
```

记录键值对，self.epoch_dict[k].append(v)，方便计算k的average / std / min / max

```python
def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
```

key (string): 记录信息的名称

val: 记录信息的值，如之前使用过store，则不需要提供（记录多值的average / std / min / max），否则需要（记录单值）

with_min_and_max (bool): 如果为真记录相应的min和max值

average_only (bool): 如果为真则只记录average值而不记录std值





## 2.2 run_utils

### 2.2.1 setup_logger_kwargs

```python
setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=False)
```

确定输出目录

1. seed为None且datastamp为false

   ```
   output_dir = data_dir/exp_name
   ```

2. seed不为NULL且datestamp为false

   ```
   output_dir = data_dir/exp_name/exp_name_s[seed]
   ```

返回：logger_kwargs ，一个字典包含output_dir和expname

## 2.3mpi

```python
def mpi_fork(n, bind_to_core=False):
```

Re-launches the current script with workers linked by MPI

n (int): Number of process to split into

bind_to_core (bool): Bind each MPI process to a core.

```python
def mpi_statistics_scalar(x, with_min_and_max=False):
```

Get mean/std and optional min/max of scalar x across MPI processes.

x: An array containing samples of the scalar to produce statistics for.

with_min_and_max (bool): If true, return min and max of x in addition to mean and std.

## 2.4执行流程

先调用
```python
logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
```
<!--logger_kwargs = dict(output_dir=，exp_name=)-->
再将logger_kwargs传递到算法vpg里

vpg算法中：
```python
logger = EpochLogger(**logger_kwargs)
logger.save_config(locals())
```

```python
logger.setup_tf_saver(sess, inputs={'x': x_ph}, outputs={'pi': pi, 'v': v})
```

每隔一定epoch保存环境状态

```python
logger.save_state({'env': env}, None)
```

每一个epoch 记录各类数据

```python
logger.log_tabular('Epoch', epoch)
logger.log_tabular('EpRet', with_min_and_max=True)
logger.log_tabular('EpLen', average_only=True)
logger.log_tabular('VVals', with_min_and_max=True)
logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
logger.log_tabular('LossPi', average_only=True)
logger.log_tabular('LossV', average_only=True)
logger.log_tabular('DeltaLossPi', average_only=True)
logger.log_tabular('DeltaLossV', average_only=True)
logger.log_tabular('Entropy', average_only=True)
logger.log_tabular('KL', average_only=True)
logger.log_tabular('Time', time.time()-start_time)
logger.dump_tabular()
```

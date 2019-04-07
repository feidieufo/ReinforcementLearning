# 1 Installation

Spinning Up 需要 Python3, OpenAI Gym, and OpenMPI

## 1.1 Installing Python

```
conda create -n spinningup python=3.6
source activate spinningup
```

需要python3.6或以上

## 1.2 Installing OpenMPI

### 1.2.1 windows

openmpi 和mpich2于2012年在win上不再更新，选择安装MSMPI。

安装MSMPI, [https://www.microsoft.com/en-us/download/details.aspx?id=57467](Microsoft MPI v10.0)

## 1.3 Installing Spinning Up

```
git clone https://github.com/openai/spinningup.git
cd spinningup
pip install -e .
```

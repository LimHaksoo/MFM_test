# 🌋 LLaVA - inspired MFM 

## Contents
- [Install](#install)
- [Dataset](https://github.com/amazon-science/spot-diff.git)
- [Train](#train)
- [Evaluation](#evaluation)

## Install


1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/LimHaksoo/MFM_test.git
cd MFM_test
```

2. Install Package
```Shell
conda create -n mfm python=3.11 -y
conda activate mfm
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Upgrade to latest code base

```Shell
git pull
pip install -e .

# if you see some import errors when you upgrade,
# please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```

## Train

Run the following shell file

```Shell
sh ~/LLaVA/scripts/v1_5/run_mfm.sh
```

## Evaluation

Run the following shell file and check the result in evaluation_log.txt

```Shell
sh ~/LLaVA/run_eval.sh
```
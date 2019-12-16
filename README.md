## Recallable meta-learning for Dialogue State Tracking

This repo is a fork of [TRADE](https://github.com/jasonwu0731/trade-dst) and [DND-LSTM](https://github.com/qihongl/dnd-lstm), including Recallable meta-learning for Dialogue State Tracking (RM-DST) and multi-task learning (MTL-DST). This code has been written using PyTorch >= 1.0. 

<p align="center">
<img src="plot/Fig1.jpg" width="75%" />
</p>

Download the MultiWOZ dataset and the processed dst version.
```console
❱❱❱ python3 create_data.py
```

## Dependency
Check the packages needed or simply run the command
```console
❱❱❱ pip install -r requirements.txt
```
If you run into an error related to Cython, try to upgrade it first.
```console
❱❱❱ pip install --upgrade cython
```

## Unseen Domain DST

#### RM-DST
Training
```console
❱❱❱ python3 myTrain_maml_DND.py -dec=TRADE -bsz=32 -dr=0.2 -lr=0.001 -le=1 -exceptd=${domain}
```

#### MTL-DST
Training
```console
❱❱❱ python3 myTrain_MTL.py -dec=TRADE -bsz=32 -dr=0.2 -lr=0.001 -le=1 -exceptd=${domain} -add_name=MTL
```
* -exceptd: except domain selection, choose one from {hotel, train, attraction, restaurant, taxi}.

#### Fine-tune

RM-DST
```console
❱❱❱ python3 fine_tune_dnd.py -bsz=8 -dr=0.2 -lr=0.001 -path=${save_path_except_domain} -exceptd=${except_domain}
```
MTL-DST
```console
❱❱❱ python3 fine_tune.py -bsz=8 -dr=0.2 -lr=0.001 -path=${save_path_except_domain} -exceptd=${except_domain}
```


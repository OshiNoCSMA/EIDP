# EIDP
This is our official implementation for the paper **Explicit and Implicit Modeling via Dual-Path Transformer for Behavior Set-informed Sequential Recommendation**, accepted by **KDD'24**
* * *

## Behavior Set-informed Sequential Recommendation (BSSR) Illustration
![](BSSR.png)
* * *

## EIDP Architecture
![](EIDP.png)
* * *

## Requirement
python==3.7.1

pytorch==1.13.1

torchvision==0.14.1

CUDA 11.7
* * *

## Command
Our implementation code has some references from the source codes of ([ICLRec](https://github.com/salesforce/ICLRec)) and ([RecBole](https://github.com/RUCAIBox/RecBole)).

For QK-Video, the execution command is,
```bash
python main.py --dataset=QKV
```

For QK-Article, the execution command is,
```bash
python main.py --dataset=QKA
```

For fast evaluation and results reproduction:
```bash
python main.py --dataset=[...] --do_eval
```
* * *

## Supplementary Materials
[A.1 Notation Table](A1NotationTable.pdf)

[A.2 ProbSparse multi-head self-attention mechanism](A2Algo_PSA.pdf)
* * *

If you have any issues or ideas, feel free to contact us ([2252271001@email.szu.edu.cn](mailto:2252271001@email.szu.edu.cn); [CNWorldisyourFC@gmail.com](mailto:CNWorldisyourFC@gmail.com)).

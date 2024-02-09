# KDD24_EIDP
Explicit and Implicit Modeling via Dual-Path Transformer for Behavior Set-informed Sequential Recommendation
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

`python main.py --dataset=QKV`

For QK-Article, the execution command is,

`python main.py --dataset=QKA`

For fast evaluation and results reproduction:

`python main.py --dataset=[...] --do_eval`

* * *
If you have any issues or ideas, feel free to contact us ([CNWorldisyourFC@gmail.com](mailto:CNWorldisyourFC@gmail.com)).

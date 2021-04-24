# ERIQG

Code implementation for the text infilling model in our work [Expanding, Retrieving and Infilling: Diversifying Cross-Domain Question Generation with Flexible Templates](https://www.aclweb.org/anthology/2021.eacl-main.279.pdf).

#### Requirements
- `python2.7`.
- `PyTorch 0.4.0`.
- `CUDA 9.0`
- The code has been tested on GTX 1080 Ti running on Ubuntu 16.04.4 LTS.

#### Training and Testing

- Pre-trained model and word embeddings can be downloaded from [here](https://github.com/xiaojingyu92/ERIQG/releases/tag/v0.1). 
- Copy eriqg_best.pth.tar  to code folder.
- Copy usedwordemb.npy to glove folder.

- To train the model by running:
 `python train.py --batch_size=16 --teacher_forcing_fraction 1.0 --prefix 'eriqg'` on terminal.

- To test on pre-trained model by running:
 `python evaluation.py --batch_size=16 --teacher_forcing_fraction 0.0 --resume eriqg_best.pth.tar` on terminal.


###  Citation

If you use ERIQG, please cite the following work:
```
@inproceedings{yu2021expanding,
  title={Expanding, Retrieving and Infilling: Diversifying Cross-Domain Question Generation with Flexible Templates},
  author={Yu, Xiaojing and Jiang, Anxiao},
  booktitle={Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume},
  pages={3202--3212},
  year={2021}
}
```

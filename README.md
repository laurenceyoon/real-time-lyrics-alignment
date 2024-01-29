## A Real-Time Lyrics Alignment System Using Chroma and Phonetic Features for Classical Singing Performance

A real-time lyrics tracking system for vocal performance.

[![arXiv](https://img.shields.io/badge/arXiv-2401.09200-b31b1b.svg)](https://arxiv.org/abs/2401.09200)

The source code of the paper [A Real-Time Lyrics Alignment System Using Chroma And Phonetic Features For Classical Vocal Performance](https://arxiv.org/abs/2401.09200), accepted by ICASSP 2024.


## Data Preparation
The dataset for evaluation is available at [here](https://github.com/laurenceyoon/winterreise_rt). Please download the dataset and put it under the `data` folder.

## Environment Setup

```bash
$ conda env create -f environment.yml
$ conda activate rt-lyrics
```

## Usage

Open the `evaluation.ipynb` notebook and run the cells.

## License

This project is under the CC-BY-NC 4.0 license. See LICENSE for details.

## Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.

```
@article{park2024real,
  title={A Real-Time Lyrics Alignment System Using Chroma And Phonetic Features For Classical Vocal Performance},
  author={Park, Jiyun and Yong, Sangeon and Kwon, Taegyun and Nam, Juhan},
  journal={arXiv preprint arXiv:2401.09200},
  year={2024}
}
```

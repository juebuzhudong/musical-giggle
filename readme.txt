# Covid19_predict

We provide here the PyTorch training script to train and test the models including ResNet18, ResNet50, VGG16, DenseNet121,DenseNet169 and EfficientNet on COVID-10 Detection.

## Environment
The code is based on Python 3.7 and PyTorch 1.3.
Running and testing on one GTX1060 6GB

## Requirements
The main requirements are listed below:

* Pytorch
* re
* skimage
* torchvision
* Python 3.7
* Numpy
* OpenCV
* Scikit-Learn

# Steps to generate the dataset used to do Training and Testing
1. Download images from the repo `Images-processed`
2. Download .txt files for image names in train, val, and test set from `Data-split` repo
3. Use the dataloader defined in line `80` of the script `DenseNet_predict.py` and load the dataset


#####  Data Description #####
This dataset mainly contains: 
1)  Public COVID-19 chest CT dataset constructed by He et al. which includes 349 COVID-19 CT scans and 397 non-COVID-19 CT images. ( https://github.com/UCSD-AI4H/COVID-CT )
2)  Other 67 CT scans of COVID-19 patients collected from local hospital.
In the end, 416 positive CT scans and 397 negative CT scans were used in this work. The following Table  shows the detail of the datasets used in this work.

| Type |       COVID-19        |      NonCOVID-19      |            |
  |  Provided by He et al.  |   Added  |  Provided by He et al.  | Added  |   Total  |
| Train |  191  | 37 |  234 |  0  |  462  |
|  Val  |  60   | 11 |  58  |  0  |  129  |
| Test  |  98   | 19 |  105 |  0  |  222  |

We provide a data split in `./Data-split`. which splits the COVID and Non_COVID samples into Train/Val/Test subsets and save them as txt. Patient distribution in each set will be updated soon.

# The public COVID-19 chest CT dataset constructed by He et al. and the six models are from https://github.com/UCSD-AI4H/COVID-CT.

**We are continuously adding new COVID CT images and we would like to invite the community to contribute COVID CTs as well.**


### Data Description

The COVID-CT-Dataset contains 349 CT images. They are in `./Images-processed/CT_COVID.zip`

Non-COVID CT scans are in `./Images-processed/CT_NonCOVID.zip`

We provide a data split in `./Data-split`.
Data split information see `README for DenseNet_predict.md`

The meta information (e.g., patient ID, patient information, DOI, image caption) is in `COVID-CT-MetaInfo.xlsx`


The images are collected from COVID19-related papers from medRxiv, bioRxiv, NEJM, JAMA, Lancet, etc. CTs containing COVID-19 abnormalities are selected by reading the figure captions in the papers. All copyrights of the data belong to the authors and publishers of these papers.

The dataset details are described in this preprint: [COVID-CT-Dataset: A CT Scan Dataset about COVID-19](https://arxiv.org/pdf/2003.13865.pdf)

If you find this dataset and code useful, please cite:

    @article{zhao2020COVID-CT-Dataset,
      title={COVID-CT-Dataset: a CT scan dataset about COVID-19},
      author={Zhao, Jinyu and Zhang, Yichen and He, Xuehai and Xie, Pengtao},
      journal={arXiv preprint arXiv:2003.13865}, 
      year={2020}
    }

### Baseline Performance
We developed a baseline method for the community to benchmark with.
The code are in the "Covid19_predict" folder and the details are in the readme files under that folder. The methods are described in [Sample-Efficient Deep Learning for COVID-19 Diagnosis Based on CT Scans](https://www.medrxiv.org/content/10.1101/2020.04.13.20063941v1)

If you find the code useful, please cite:

    @Article{he2020sample,
      author  = {He, Xuehai and Yang, Xingyi and Zhang, Shanghang, and Zhao, Jinyu and Zhang, Yichen and Xing, Eric, and Xie,       Pengtao},
      title   = {Sample-Efficient Deep Learning for COVID-19 Diagnosis Based on CT Scans},
      journal = {medrxiv},
      year    = {2020},
    }


### Contribution Guide
 - To contribute to our project, please email your data to jiz077@eng.ucsd.edu with the corresponding meta information (Patient ID, DOI and Captions).
 - We recommend you also extract images from publications or preprints. Make sure the original papers you crawled have different DOIs from those listed in `COVID-CT-MetaInfo.xlsx`.
 - In `COVID-CT-MetaInfo.xlsx`, images with the form of `2020.mm.dd.xxxx` are crawled from bioRxiv or medRxiv. The DOIs for these preprints are `10.1101/2020.mm.dd.xxxx`.
 
To communicate with us, please email us by  2462473897@qq.com, 517581496@qq.com, 237916164@qq.com, or 717434157@qq.com.
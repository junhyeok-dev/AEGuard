# AEGuard
## Motivation

As computer vision technology and image-based machine learning have developed and widespread, massive security systems based on image data are designed and deployed. At the same time, adversarial attack, the security attack that makes the image-based machine learning model misjudge by modulating input image, also has developed and widespread.

AEGuard is an adversarial sample detection model based on edge noise features, which commonly appears in adversarial samples.

## Tech/framework used

* Tensorflow
* OpenCV

## Requirements

* Python 3
  * IPython 7.18.1 or greater
  * Tensorflow 2.3.0 or greater
  * Scikit-image 0.17.2 or greater
  * Scikit-learn 0.23.2 or greater
  * Numpy 1.18.5 or greater
  * Pandas 1.1.3 or greater
* OpenCV 4

## Features

AEGuard detects adversarial sample.

## Installation

### Linux(Ubuntu)

```bash
#Install Python3 and Git
sudo apt install python3 python3-pip git

#Install requirements (You can use tensorflow-gpu instead of tensorflow if you have CUDA-supported GPU)
pip3 install jupyter tensorflow scikit-learn scikit-image numpy pandas

#Download AEGuard source
git clone https://github.com/junhyeok-dev/AEGuard.git
```

## How to use?



## Performance

AEGuard shows 90.1043% detection accuracy for JSMA (e=0.1) adversarial samples. It is 12.94% higher than Grosse et al. (2018)
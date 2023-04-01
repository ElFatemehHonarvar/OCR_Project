# CRAFT-train

On the official CRAFT github, there are many people who want to train CRAFT models. 

However, the training code is not published in the official CRAFT repository. 

There are other reproduced codes, but there is a gap between their performance and performance reported in the original paper. (https://arxiv.org/pdf/1904.01941.pdf) 

The trained model with this code recorded a level of performance similar to that of the original paper.

```bash
├── config
│   ├── syn_train.yaml
│   └── custom_data_train.yaml
├── data
│   ├── pseudo_label
│   │   ├── make_charbox.py
│   │   └── watershed.py
│   ├── boxEnlarge.py
│   ├── dataset.py
│   ├── gaussian.py
│   ├── imgaug.py
│   └── imgproc.py
├── loss
│   └── mseloss.py
├── metrics
│   └── eval_det_iou.py
├── model
│   ├── craft.py
│   └── vgg16_bn.py
├── utils
│   ├── craft_utils.py
│   ├── inference_boxes.py
│   └── utils.py
├── trainSynth.py
├── train.py
├── train_distributed.py
├── eval.py
├── data_root_dir   (place dataset folder here)
└── exp             (model and experiment result files will saved here)
```

### Installation

Install using `pip`

``` bash
pip install -r requirements.txt
```


### Training

1. Put your training, test data in the following format

   ```
   └── data_root_dir (you can change root dir in yaml file)
       ├── ch4_training_images
       │   ├── img_1.jpg
       │   └── img_2.jpg
       ├── ch4_training_localization_transcription_gt
       │   ├── gt_img_1.txt
       │   └── gt_img_2.txt
       ├── ch4_test_images
       │   ├── img_1.jpg
       │   └── img_2.jpg
       └── ch4_training_localization_transcription_gt
           ├── gt_img_1.txt
           └── gt_img_2.txt
   ```

   * localization_transcription_gt files format :

   ```
    377,117,463,117,465,130,378,130,Genaxis Theatre
    493,115,519,115,519,131,493,131,[06]
    374,155,409,155,409,170,374,170,###
   ```

2. Write configuration in yaml format (example config files are provided in `config` folder.)

   * To speed up training time with multi-gpu, set num_worker > 0   

3. Put the yaml file in the config folder

4. Run training script like below (If you have multi-gpu, run train_distributed.py)

5. Then, experiment results will be saved to ```./exp/[yaml]``` by default.

* Step 1 : To train CRAFT with SynthText dataset from scratch

  * Note : This step is not necessary if you use <a href="https://drive.google.com/file/d/1enVIsgNvBf3YiRsVkxodspOn55PIK-LJ/view?usp=sharing">this pretrain</a> as a checkpoint when start training step 2. You can download and put it in `exp/CRAFT_clr_amp_29500.pth` and change `ckpt_path` in the config file according to your local setup.

  ```
  CUDA_VISIBLE_DEVICES=0 python3 trainSynth.py --yaml=syn_train
  ```

* Step 2 : To train CRAFT with [SynthText + IC15] or custom dataset

  ```
  CUDA_VISIBLE_DEVICES=0 python3 train.py --yaml=custom_data_train               ## if you run on single GPU
  CUDA_VISIBLE_DEVICES=0,1 python3 train_distributed.py --yaml=custom_data_train   ## if you run on multi GPU
  ```

### Arguments

* ```--yaml``` : configuration file name

### Evaluation

* In the official repository issues, the author mentioned that the first row setting F1-score is around 0.75.
* In the official paper, it is stated that the result F1-score of the second row setting is 0.87.
  * If you adjust post-process parameter 'text_threshold' from 0.85 to 0.75, then F1-score reaches to 0.856.
* It took 14h to train weak-supervision 25k iteration with 8 RTX 3090 Ti.
  * Half of GPU assigned for training, and half of GPU assigned for supervision setting.

| Training Dataset      | Evaluation Dataset | Precision | Recall | F1-score |                                             pretrained model |
| --------------------- | ------------------ | :-------: | :----: | :------: | -----------------------------------------------------------: |
| SynthText             | ICDAR2013          |   0.801   | 0.748  |  0.773   | <a href="https://drive.google.com/file/d/1enVIsgNvBf3YiRsVkxodspOn55PIK-LJ/view?usp=sharing">download link</a> |
| SynthText + ICDAR2015 | ICDAR2015          |   0.909   | 0.794  |  0.848   | <a href="https://drive.google.com/file/d/1qUeZIDSFCOuGS9yo8o0fi-zYHLEW6lBP/view">download link</a> |



## My Experience

In this document, I want to train and detect my custom dataset on a CRAFT model. Before the training process begins, there are some challenges and solutions related to my dataset that I aim to discuss here.

### - Dataset Preparation

-------

Most of the time good results can be obtained with no changes to the models or training settings provided your dataset is sufficiently large, varied, and well-labeled. Unfortunately, my custom dataset does not satisfy these conditions, and I want to describe how I met some of these challenges.

1. The initial label format was as follow:

   9 20 36 54
   37 19 54 55
   55 19 81 54
   82 20 98 55
   99 20 124 54
   125 19 138 55
   139 19 168 53

   The numbers represent the coordinates of the bounding boxes around each character in the generated synthetic text images, but according to EasyOCR instructions, the 8 numbers for each coordination as well as the text content in the image should be mentioned:

    9,55,168,55,168,19,9,19,D C B A

2. With respect to the suggested name for the label on Github, I made a change to my dataset TXT label from   ***task_data_0_boxes.txt***  to ***gt_task_data_0.txt***

3. The other challenge is that Persian contents were reversed:

   فیدر     <-   ردیف

4. Let's take a look at a sample:

   ![image-20230329223412075](/home/elham/.config/Typora/typora-user-images/image-20230329223412075.png)

   As you can see the Persian text is not defined in OpenCV.

5. You can find the data preparation and analysis process *OCR_Custom_Dataset.ipynb*

6.  Training and detection results are available in  *OCR_Training_Results.ipynb*

   

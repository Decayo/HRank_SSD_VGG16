# SSD-VGG16 Use HRank Method
## 摘要
&emsp;&emsp;利用HRank方法剪裁ssd-vgg16模型，剪裁大小至mobilev1並比較之間的準確率(mAP)、執行時間(FPS)、各class的準確度、FLOPs和Params。

## 如何使用？
推薦執行colab程式，可直接執行，附帶預訓練模型、資料及下載。  
包含評估以及訓練(可選擇)，2012/6/24確認可完全執行。  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yMJ0hbWdZbCk3RgYCB4abHL_J-bzFTzr?usp=sharing)  
(Include English and Chinese commit and pre-trained model)  


## 預訓練和rank資訊下載  
(https://drive.google.com/file/d/1g2KkOMTeZ7u_EM9Ke2HpomvHm049Iqit/view?usp=sharing) (含有剪裁完成、原VGG-16、mobile-netv1)
- [SSD-vgg-16 rank資訊](https://drive.google.com/drive/folders/1_qmJN0YcOlD7iuxoy7PXnUh3EH7LZwtj?usp=sharing)
## 資料集
- [VOC Mirror](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)
 
## 訓練設置
### 1. 先對預訓練模型分解出每層的rank資訊  
``` 
 python HR_rank_generation.py --dataset VOCdevkit/VOC2012/
```  
將會存於```rank_conv/vgg-16-limitbatch5/```  
### 2. 接著開始訓練

```
python  train_ssd_prune_eval.py  --datasets VOCdevkit/VOC2012/ --validation_dataset VOCdevkit/VOC2007-test/ --base_net models/vgg16-ssd-mp-0_7726.pth --batch_size 8 --num_epochs 120 --lr 1e-5 --scheduler cosine --label_file models/voc-model-labels.txt --compress_rate [0.1]*7+[0.5]*9
```
參數解釋：
``` 
--datasets           訓練集位置，可複數個(VOCdevkit/VOC2012/ VOCdevkit/VOC2007/)
--validation_dataset 驗證集，預設為voc2007之測試集
--base_net           原始育訓練集，第一次訓練需要此參數
-batch_size          批次大小
--num_epochs         訓練Epoch，該實驗於120epoch,lr:1e-5收斂完成
--lr                 學習率，預設為1e-5
--scheduler          學習率規劃方法：提供cosine與Multuistep方法，預設為cosine
--label_file         欲訓練clas，預設位於 models/ 內，20個classes
--compress_rate      模型壓縮率，vgg16為16層需要給予每一層壓縮率
```
從check point繼續訓練：
``` 
python  train_ssd_prune_eval.py  --datasets VOCdevkit\VOC2012\  --validation_dataset VOCdevkit\VOC2007-test\ --resume models/model_best.pth-130.tar --batch_size 8 --num_epochs 300 --lr 1e-5 --scheduler cosine --label_file models\voc-model-labels.txt --compress_rate [0.1]*7+[0.5]*5+[0.5]*4 
```
參數解釋：
```
--resume            訓練過程中產生之models
--base_net          斷點接續訓練不需此參數

其他注意事項：
因參數需求不同，接續訓練的模型格式為： 
model_best.pth-{Epock}.tar
此模型僅提供於訓練，因內含除了state_dict以外的參數因此檔案較大。
若需後續使用請使用：
vgg16-ssd-Epoch-{Epoch}-Loss-{Validation Loss}.pth
該格式之模型為最終輸出，請挑選loss最小的模型做為您的評估和影片輸出。
```
### 3.評估mAP
```
python eval_ssd.py   --dataset VOCdevkit/VOC2007-test  --trained_model models/vgg16-ssd-mp-0_7726.pth --label_file models/voc-model-labels.txt  --compress_rate [0.]*100
```
參數解釋：
```
--trained_model   欲評估之模型路徑
--compress_rate   欲評估之壓縮率
```
### 4.輸出SSD影片結果
```
python run_ssd_video_demo.py vgg16-ssd models/vgg16-ssd-mp-0_7726.pth models/voc-model-labels.txt test_vid/bottle.mp4 output_vid/b_ORI_ssd.mp4
```

參數解釋：
```
依照順序進行輸入：
1. 模型路徑
2. 欲偵測之label文字檔路徑
3. 輸入video路徑和名稱
4. 輸出video路徑和名稱
```
## 評估結果
### 1. 預測的準確率與平均預測時間  
環境：Nvidia Tesla T4 on colab
#### VGG-16
```
Avg predictime : 0.026228126012065683


Average Precision Per-class:
aeroplane: 0.8025462619286368
bicycle: 0.8299833173214867
bird: 0.7576572118665351
boat: 0.7045265858094326
bottle: 0.5158965984153355
bus: 0.8357903732556332
car: 0.859542957744215
cat: 0.8703022563599616
chair: 0.617540968338572
cow: 0.8076502429161613
diningtable: 0.7620883148957985
dog: 0.8441789472265441
horse: 0.8691892887075761
motorbike: 0.849310900730183
person: 0.7933838111856593
pottedplant: 0.5274780629633236
sheep: 0.7780796261366258
sofa: 0.8029172822438873
train: 0.8703971417174846
tvmonitor: 0.7640327994994804

Average Precision Across All Classes:0.7731246474631266
```
#### Pruned VGG-16
```
Avg predictime : 0.04258417543956652


Average Precision Per-class:
aeroplane: 0.7261154830357192
bicycle: 0.772867977029674
bird: 0.6702559884805529
boat: 0.6192405135444621
bottle: 0.4026901906395809
bus: 0.8012662751921901
car: 0.8197566578626294
cat: 0.8526189542830351
chair: 0.5208981418671049
cow: 0.6874637716090424
diningtable: 0.6708198351046444
dog: 0.7582606006551614
horse: 0.8020379526622873
motorbike: 0.7698874182317722
person: 0.7397587122237995
pottedplant: 0.3949796299586133
sheep: 0.6816220550234936
sofa: 0.7354843706723636
train: 0.7832249203603624
tvmonitor: 0.6973618672614345

Average Precision Across All Classes:0.6953305657848962
```
#### MobileNetV1
```
Avg predictime : 0.02493384143454963


Average Precision Per-class:
aeroplane: 0.6843271224059599
bicycle: 0.7911140237662206
bird: 0.6171819168583986
boat: 0.5612220055063379
bottle: 0.3485216621466003
bus: 0.7677814849265677
car: 0.7280986468467315
cat: 0.8369208203985581
chair: 0.5169138632991064
cow: 0.6238697603075337
diningtable: 0.7062172972736019
dog: 0.7872656014540705
horse: 0.819446325939355
motorbike: 0.7918539457195842
person: 0.702363739134837
pottedplant: 0.3985294933564374
sheep: 0.6066678298227772
sofa: 0.7573083661544429
train: 0.8262441264750008
tvmonitor: 0.6461898726506375

Average Precision Across All Classes:0.6759018952221378
```
### 2. FLOPs & Paras
#### VGG-16
```
compress rate:  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
0.0 [64, 3, 3, 3] params: 1792.0  flops: 155520000.0
0.0 [64, 64, 3, 3] params: 36928.0  flops: 3317760000.0
0.0 [128, 64, 3, 3] params: 73856.0  flops: 1658880000.0
0.0 [128, 128, 3, 3] params: 147584.0  flops: 3317760000.0
0.0 [256, 128, 3, 3] params: 295168.0  flops: 1658880000.0
0.0 [256, 256, 3, 3] params: 590080.0  flops: 3317760000.0
0.0 [256, 256, 3, 3] params: 590080.0  flops: 3317760000.0
0.0 [512, 256, 3, 3] params: 1180160.0  flops: 1703411712.0
0.0 [512, 512, 3, 3] params: 2359808.0  flops: 3406823424.0
0.0 [512, 512, 3, 3] params: 2359808.0  flops: 3406823424.0
Conv2dNo cp rate
Conv2dNo cp rate
0.0 [512, 512, 3, 3] params: 2359808.0  flops: 851705856.0
0.0 [512, 512, 3, 3] params: 2359808.0  flops: 851705856.0
0.0 [512, 512, 3, 3] params: 2359808.0  flops: 851705856.0
0.0 [1024, 512, 3, 3] params: 4719616.0  flops: 3968335872.0
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Params: 19434304.00
Flops: 31784832000.00
```
#### Pruned VGG-16
```
compress rate:  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
0.1 [57, 3, 3, 3] params: 1456.0  flops: 138510000.0
0.1 [57, 57, 3, 3] params: 26728.0  flops: 2631690000.0
0.1 [115, 57, 3, 3] params: 53456.0  flops: 1327387500.0
0.1 [115, 115, 3, 3] params: 107744.0  flops: 2678062500.0
0.1 [230, 115, 3, 3] params: 214452.0  flops: 1339031250.0
0.1 [230, 230, 3, 3] params: 428697.0  flops: 2678062500.0
0.1 [230, 230, 3, 3] params: 428697.0  flops: 2678062500.0
0.5 [256, 230, 3, 3] params: 265088.0  flops: 765204480.0
0.5 [256, 256, 3, 3] params: 295040.0  flops: 851705856.0
0.5 [256, 256, 3, 3] params: 295040.0  flops: 851705856.0
Conv2dNo cp rate
Conv2dNo cp rate
0.5 [256, 256, 3, 3] params: 295040.0  flops: 212926464.0
0.5 [256, 256, 3, 3] params: 295040.0  flops: 212926464.0
0.5 [256, 256, 3, 3] params: 295040.0  flops: 212926464.0
0.5 [512, 256, 3, 3] params: 590080.0  flops: 992083968.0
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Conv2dNo cp rate
Params: 3591598.00
Flops: 14224900401.00
```
#### MobileNetV1
```
Params: 4231976.00
Flops: 755703808.00
```


3. demo預覽  
![](https://i.imgur.com/DveaYbM.jpg)
![](https://i.imgur.com/KvsymOh.jpg)
![](https://i.imgur.com/2sWvkW4.jpg)

## 參考資料
https://github.com/qfgaohao/pytorch-ssd  
https://github.com/lmbxmu/HRankPlus


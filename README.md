# PillaTech_team04
코드잇 스프린트 AI 9기 4팀 1차 프로젝트
- 베이스라인 1.0버전(실험환경 재현x) 

## 문제 상황 
- 공식 베이스라인 1.0을 git에 세팅하기 전,  동일환경 재현성 테스트를 했는데 한별님이 캐글 대시보드에 올린 0.70점대와 다른 0.9점대가 kaggle에서 2번 나옴
- 0.7점대와 0.9점대는 차이가 너무 크기 때문에, 공식 baseline 1.0 코드와 환경설정 자료를 git에 세팅하기 어려운 상황
- 당시에는 각자 자유롭게 실험을 하는 목적이었어서 재현성을 위한 기록을 많이 하지 않았음

## 문제 원인 
- 초기에 각자 실험을 하다 보니 실험 재현성을 위한 세분화된 조건들을 설정해두지 않고 진행했고, mac, window환경이 다른 데 제출용 컴퓨터를 정하지 않고 각자 로컬 컴퓨터 실험결과를 공식 kaggle에 올려서 그런 듯 함

## 참고
- default라고 써있는건 내 컴퓨터의 default값인 것으로 추측됨.
- 같은 yolov8n을 써도 컴퓨터마다 실험 결과가 달라질 수 있나?
    - **그럴 수 있음**. 보통 원인은 모델명이 아니라
    - ultralytics 버전 차이
    - 스크립트/CLI/YAML override 차이
    - 환경(torch/cuda) 차이
- 언제 같아지나?
    - ultralytics 버전 + 코드 + YAML + 의존성이 같으면 기본값/동작은 사실상 동일

experiments.md파일 
```
# 1. train yaml파일 설정 
```
# 3/28일에 1.0train시킬 때 설정 값 
name: "exp15_train_baseline_yolov8n_1.0"
model: "yolov8n.pt"
data: "data/yolo_dataset/dataset.yaml"
epochs: 50 
imgsz: 640 
batch: 16
workers: 8 # default
patience: 100 # default
amp: true # default
close_mosaic: 10 # default

# Optimization
# yolov8n default는 auto이며 auto설정이면 sgd나 adamw 둘 중 하나가 선택된다고 함.  
optimizer: "auto" # default ->내 컴퓨터에서는 interation이 100000이하면 AdamW으로 사용된다고 설정되어 있었음  
lr0: 0.01 # default
lrf: 0.01 # default
momentum: 0.937 # default
weight_decay: 0.0005 # default
warmup_epochs: 3.0 # default
warmup_momentum: 0.8 # default
warmup_bias_lr: 0.1 # default
cos_lr: true # default
pretrained: true # default
resume: false # default

# Augmentation  
hsv_h: 0.015 # default
hsv_s: 0.7 # default
hsv_v: 0.4 # default
translate: 0.1 # default
scale: 0.5 # default
shear: 0.0 # default
perspective: 0.0 # default
fliplr: 0.5 # default
flipud: 0.0 # default
degrees: 0.0 # default
mosaic: 1.0 # default
copy_paste: 0.0 # default
mixup: 0.0 # default
erasing: 0.4 # default
auto_augment: "randaugment" # default

# 재현성 보장
seed: 42
deterministic: true # default

# Device
device: "0" # default
```

# 2. inference yaml 
```
model: "runs/exp14_baseline_yolov8n_1.0/weights/best.pt"
imgsz: 640
conf: 0.25
iou: 0.70
output: "submission/exp14_baseline_yolov8n_1.0.csv"
test_images: "data/raw/sprint_ai_project1_data/test_images"
data: "data/yolo_dataset/dataset.yaml"
json_dir: "data/raw/sprint_ai_project1_data/train_annotations"
save_config: true
```

# 3. args.yaml
```
task: detect
mode: train
model: yolov8n.pt
data: data/yolo_dataset/dataset.yaml
epochs: 50
time: null
patience: 100
batch: 16
imgsz: 640
save: true
save_period: -1
cache: false
device: '0'
workers: 8
project: /PillaTech_team04/runs
name: exp14_train_baseline_yolov8n_1.0
exist_ok: false
pretrained: true
optimizer: auto
verbose: true
seed: 42
deterministic: true
single_cls: false
rect: false
cos_lr: true
close_mosaic: 10
resume: false
amp: true
fraction: 1.0
profile: false
freeze: null
multi_scale: 0.0
compile: false
overlap_mask: true
mask_ratio: 4
dropout: 0.0
val: true
split: val
save_json: false
conf: null
iou: 0.7
max_det: 300
half: false
dnn: false
plots: true
end2end: null
source: null
vid_stride: 1
stream_buffer: false
visualize: false
augment: false
agnostic_nms: false
classes: null
retina_masks: false
embed: null
show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false
show_labels: true
show_conf: true
show_boxes: true
line_width: null
format: torchscript
keras: false
optimize: false
int8: false
dynamic: false
simplify: true
opset: null
workspace: null
nms: false
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 7.5
cls: 0.5
dfl: 1.5
pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
nbs: 64
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
bgr: 0.0
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
copy_paste_mode: flip
auto_augment: randaugment
erasing: 0.4
cfg: null
tracker: botsort.yaml
save_dir: /PillaTech_team04/runs/exp14_train_baseline_yolov8n_1.0
```
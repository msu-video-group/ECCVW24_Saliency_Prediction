# ECCV-AIM Video Saliency Prediction Challenge 2024

[![Page](https://img.shields.io/badge/Challenge-Page-blue)](https://challenges.videoprocessing.ai/challenges/video-saliency-prediction.html)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-brightgreen)](https://drive.google.com/drive/folders/1Ma6xoVocgQkcnvXFAiwNoq7MfuDF-SgE?usp=sharing)
[![Dataset HuggingFace](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/)
[![Challenges](https://img.shields.io/badge/Challenges-AIM%202024-orange)](https://cvlai.net/aim/2024/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-VideoProcessing-purple)](https://videoprocessing.ai/benchmarks/)


## Dataset

We provide a novel audio-visual mouse saliency (<em>AViMoS</em>) dataset with the following key-features:
* Diverse content: movie, sports, live, vertical videos, etc.;
* Large scale: **1500** videos with mean **19s** duration;
* High resolution: all streams are **FullHD**;
* **Audio** track saved and played to observers;
* Mouse fixations from **>5000** observers (**>70** per video);
* License: **CC-BY**;

File structure:
1) `Videos.zip` - 1500 (1000 Train + 500 Test) .mp4 video (kindly reminder: many videos contain an audio stream and users watched the video with the sound turned ON!) 

2) `TrainTestSplit.json` — in this JSON we provide Train/Public Test/Private Test split of all videos 

3) `SaliencyTrain.zip'/'SaliencyTest.zip` — almost losslessly (crf 0, 10bit, min-max normalized) compressed continuous saliency maps videos for Train/Test subset 

4) `FixationsTrain.zip'/'FixationsTest.zip` — contains the following files for Train/Test subset: 

* `.../video_name/fixations.json` — per-frame fixations coordinates, from which saliency maps were obtained, this JSON will be used for metrics calculation

* `.../video_name/fixations/` — binary fixation maps in '.png' format (since some fixations could share the same pixel, this is a lossy representation and is NOT used either in calculating metrics or generating Gaussians, however, we provide them for visualization and frames count checks)

5) `VideoInfo.json` — meta information about each video (e.g. license)

## Evaluation

### Environment setup

```
conda create -n saliency python=3.8.16
conda activate saliency
pip install numpy==1.24.2 opencv-python==4.7.0.72 tqdm==4.65.0
conda install ffmpeg=4.4.2 -c conda-forge
```
### Run evaluation
Archives with videos were accepted from challenge participants as submissions and scored using the same pipeline as in `bench.py`.

Usage example:

1) Check that your predictions match the structure and names of the [baseline CenterPrior submission](https://drive.google.com/file/d/1rPgMdb4L79OD2vvpDQyqWZIDox78rmxG/view)
2) Install `pip install -r requirments.txt`, `conda install ffmpeg`
3) Download and extract `SaliencyTest.zip`,  `FixationsTest.zip`, and `TrainTestSplit.json` files from the dataset page
4) Run `python bench.py` with flags:
* `--model_video_predictions ./SampleSubmission-CenterPrior` — folder with predicted saliency videos
* `--model_extracted_frames ./SampleSubmission-CenterPrior-Frames` — folder to store prediction frames (should not exist at launch time), requires ~170 GB of free space
* `--gt_video_predictions ./SaliencyTest/Test` — folder from dataset page with gt saliency videos
* `--gt_extracted_frames ./SaliencyTest-Frames` — folder to store ground-truth frames (should not exist at launch time), requires ~170 GB of free space
* `--gt_fixations_path ./FixationsTest/Test` — folder from dataset page with gt saliency fixations
* `--split_json ./TrainTestSplit.json` — JSON from dataset page with names splitting
* `--results_json ./results.json` — path to the output results json
* `--mode public_test` — public_test/private_test subsets
5) The result you get will be available following `results.json` path


## Challenge Leaderboard

Please follow the paper to learn about the team's solutions, and challenge page for more results. 

Here we only provide the final leaderboard:

| Team Name       | AUC-Judd | CC    | SIM   | NSS   | Rank | #Params (M) |
|-----------------|:-----------:|:--------:|:---------:|:---------:|:--------:|:--------------:|
| CV_MM           | **0.894** | **0.774** | **0.635** | **3.464** | 1.00 | 420.5      |
| VistaHL         | <ins>0.892</ins> | <ins>0.769</ins> | <ins>0.623</ins> | 3.352 | 2.75 | 187.7      |
| PeRCeiVe Lab    | 0.857 | <em>0.766</em> | 0.610 | <ins>3.422</ins> | 3.75 | 402.9      |
| SJTU-MML        | 0.858 | 0.760 | <em>0.615</em> | 3.356 | 4.00 | 1288.7     |
| MVP             | 0.838 | 0.749 | 0.587 | <em>3.404</em> | 5.00 | 99.6       |
| ZenithChaser    | <em>0.869</em> | 0.606 | 0.517 | 2.482 | 5.50 | 0.19       |
| Exodus          | 0.861 | 0.599 | 0.510 | 2.491 | 6.00 | 69.7       |
| Baseline (CP)   | 0.833 | 0.449 | 0.424 | 1.659 | 8.00 | -          |

## 
## Citation

Please cite the paper if you find challenge materials useful for your research:

`@article{
}
`

from multiprocessing.pool import Pool, ThreadPool
from os import path, listdir, mkdir
from pathlib import Path
from tqdm import tqdm
from glob import glob
import numpy as np
import subprocess
import argparse
import warnings
import json
import cv2

cv2.setNumThreads(0)
eps = np.finfo(np.float32).eps
warnings.filterwarnings("error")

###metrics###

def nss(s_map, gt):
    s_map_norm = (s_map - np.mean(s_map))/(np.std(s_map) + 1e-7)
    temp = s_map_norm[gt[:, 0], gt[:, 1]]
    return np.mean(temp)


def similarity(s_map, gt):
    s_map = s_map / (np.sum(s_map) + 1e-7)
    gt = gt / (np.sum(gt) + 1e-7)
    return np.sum(np.minimum(s_map, gt))


def cc(s_map, gt):
    a = (s_map - np.mean(s_map))/(np.std(s_map) + 1e-7)
    b = (gt - np.mean(gt))/(np.std(gt) + 1e-7)
    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum() + 1e-7)
    return r


def auc_judd(S, F):
    
    Sth = S[F[:, 0], F[:, 1]]
    Nfixations = len(Sth)
    Uniqe_fixations = np.unique(F, axis=1).shape[-1]
    Possible_fixations = np.prod(S.shape) + (Nfixations - Uniqe_fixations)

    allthreshes = np.sort(Sth)[::-1]
    tp = np.zeros(Nfixations + 2)
    fp = np.zeros(Nfixations + 2)
    tp[0] = fp[0] = 0
    tp[-1] = fp[-1] = 1

    # Vectorized computation of aboveth
    aboveth = np.sum(S >= allthreshes[:, np.newaxis, np.newaxis], axis=(1, 2))

    arange = np.arange(1, Nfixations + 1)
    fp[1:-1] = (aboveth - arange) / (Possible_fixations - Nfixations)
    tp[1:-1] = arange / Nfixations

    # Trapezoidal integration to compute AUC-Judd
    return np.trapz(tp, fp)



def kldiv(s_map, gt):
    s_map = s_map / (np.sum(s_map) * 1.0)
    gt = gt / (np.sum(gt) * 1.0)
    eps = 2.2204e-16
    res = np.sum(gt * np.log(eps + gt / (s_map + eps)))
    return res


######

def xrgb2gray(img):
    assert len(img.shape) in (2, 3)
    return img.mean(axis=2) if len(img.shape) == 3 else img

# Returns SM in [0; 1] range
def read_sm(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = xrgb2gray(img)
    img = (img - img.min()) / (img.max() - img.min() + eps)
    return img

def calculate_frame_metrics(frame):
    gt_fix = np.array(frame['gt_fixations'])
    gt_120_sm = read_sm(frame['gt_saliency_path'])
    pred_sm = cv2.resize(read_sm(frame['predictions_path']), (gt_120_sm.shape[1], gt_120_sm.shape[0]))

    return {
        'sim_score': similarity(pred_sm, gt_120_sm),
        'nss_score': nss(pred_sm, gt_fix),
        'cc_score': cc(pred_sm, gt_120_sm),
        'auc_judd_score': auc_judd(pred_sm, gt_fix),
    }


def calculate_metrics(video_name, temp_predictions_path, temp_gt_saliency_path, temp_gt_fixations_path, num_workers=4):
    predictions_path = glob(temp_predictions_path)[0]
    gt_saliency_path = glob(temp_gt_saliency_path)[0]
    with open(temp_gt_fixations_path) as f:
        gt_fixations = json.load(f)

    scores = []
    assert_func = lambda path: set([int(x.split('.')[0]) for x in listdir(path)])
    assert assert_func(gt_saliency_path) == assert_func(predictions_path)

    frames = [
        {
            'gt_fixations': gt_fix,
            'gt_saliency_path': gt_sal,
            'predictions_path': pred,
        } for gt_fix, gt_sal, pred in zip(
            gt_fixations,
            [path.join(gt_saliency_path, x) for x in sorted(listdir(gt_saliency_path))],
            [path.join(predictions_path, x) for x in sorted(listdir(predictions_path))]
    )]
    with Pool(num_workers) as pool:
        scores = pool.map(calculate_frame_metrics, frames)

    conv_scores = {metric: [x[metric] for x in scores] for metric in scores[0].keys()}

    return {
        'video_name' : video_name,
        'cc' : np.mean(conv_scores['cc_score']),
        'sim' : np.mean(conv_scores['sim_score']),
        'nss' : np.mean(conv_scores['nss_score']),
        'auc_judd' : np.mean(conv_scores['auc_judd_score']),
    }


def calculate_all_videos(video_names, model_extracted_frames, gt_extracted_frames, gt_fixations_path, num_workers=4):
    
    detail_result = []
    for video_name in tqdm(video_names):
        if len([x for x in detail_result if x['video_name'] == video_name]) > 0:
            continue
        short_video_name = Path(video_name).name
        model_output = str(Path(model_extracted_frames) / f'{short_video_name}*')
        gt_gaussians = str(Path(gt_extracted_frames) / f'{short_video_name}*')
        gt_fixations = Path(gt_fixations_path) / short_video_name / 'fixations.json'
        cur_result = calculate_metrics(video_name, model_output, gt_gaussians, gt_fixations, num_workers)
        detail_result += [cur_result]
        np.save("tmp2.npy", detail_result)

    return detail_result


def make_bench(model_extracted_frames, gt_extracted_frames, gt_fixations_path, split_json='TrainTestSplit.json', results_json='results.json', mode='public_test', num_workers=4):

    print(num_workers, 'worker(s)')
    print(f'Testing {model_extracted_frames}')

    sm_listdir = listdir(model_extracted_frames)
    gt_listdir = listdir(gt_extracted_frames)

    if len(sm_listdir) < len(gt_listdir):
        msg = f'There are results for only a few videos ({len(sm_listdir)}/{len(gt_listdir)})!'
        raise ValueError(msg)

    video_names = sorted(sm_listdir)
    with open(split_json) as f:
        splits = set(json.load(f)[mode])

    video_names = [name for name in video_names if name in splits]

    detail_result = calculate_all_videos(video_names, model_extracted_frames, gt_extracted_frames, gt_fixations_path, num_workers)
    detail_result = sorted(detail_result, key=lambda res: res['video_name'])

    result = {'cc' : [], 'sim' : [], 'nss' : [], 'auc_judd' : []}
    for i in result:
        for j in detail_result:
            result[i].append(j[i])

    with open(results_json, 'w') as f:
        json.dump(result, f)

    model_res = {'Model': [model_extracted_frames], 'Mode': [mode]}
    [model_res.update({key: [np.mean(result[key])]}) for key in result.keys()]
    
    print(model_res)



def extract_frames(input_dir, output_dir, split_json='TrainTestSplit.json', mode='public_test', num_workers=4):

    def poolfunc(x):
        if x.stem not in splits[mode]:
            return
        dst_vid = dst / x.stem
        if dst_vid.exists():
            pbar.update(1)
            return
        dst_vid.mkdir()
        subprocess.check_call(f'ffmpeg -v error -i {x} {dst_vid}/%03d.png'.split())
        pbar.update(1)

    with open(split_json) as f:
        splits = json.load(f)

    root = Path(input_dir)
    dst = Path(output_dir)
    dst.mkdir(exist_ok=True)
    videos = list(root.iterdir())
    pbar = tqdm(total=len(splits[mode]))
    with ThreadPool(num_workers) as p:
        p.map(poolfunc, videos)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_video_predictions', default='./SampleSubmission-CenterPrior',
                                help='Folder with predicted saliency videos')
    parser.add_argument('--model_extracted_frames', default='./SampleSubmission-CenterPrior-Frames', 
                                help='Folder to store prediction frames (should not exist at launch time), requires ~170 GB of free space')

    parser.add_argument('--gt_video_predictions', default='./SaliencyTest/Test',
                                help='Folder from dataset page with gt saliency videos')
    parser.add_argument('--gt_extracted_frames', default='./SaliencyTest-Frames', 
                                help='Folder to store ground-truth frames (should not exist at launch time), requires ~170 GB of free space')
    parser.add_argument('--gt_fixations_path', default='./FixationsTest/Test',
                                help='Folder from dataset page with gt saliency fixations')
    parser.add_argument('--split_json', default='./TrainTestSplit.json',
                                help='Json from dataset page with names splitting')

    parser.add_argument('--results_json', default='./results.json')
    parser.add_argument('--mode', default='public_test', help='public_test/private_test')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    if not path.exists(args.model_extracted_frames):
        print("Extracting", args.model_video_predictions, 'to', args.model_extracted_frames)
        extract_frames(args.model_video_predictions, args.model_extracted_frames, args.split_json, args.mode, args.num_workers)
    if not path.exists(args.gt_extracted_frames):
        print("Extracting", args.gt_video_predictions, 'to', args.gt_extracted_frames)
        extract_frames(args.gt_video_predictions, args.gt_extracted_frames, args.split_json, args.mode, args.num_workers)

    make_bench(args.model_extracted_frames, args.gt_extracted_frames, args.gt_fixations_path, args.split_json, args.results_json, args.mode, args.num_workers)

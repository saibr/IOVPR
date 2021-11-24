from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import numpy as np
import faiss
import os.path
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from guppy import hpy
from .pca import PCA
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils.dist_utils import synchronize
from .utils.serialization import write_json
from .utils.data.preprocessor import Preprocessor
from .utils import to_torch
import pickle
import psutil
import gc


def extract_cnn_feature(model, inputs, vlad=True, gpu=None):
    model.eval()
    inputs = to_torch(inputs).cuda(gpu)
    outputs = model(inputs)
    if (isinstance(outputs, list) or isinstance(outputs, tuple)):
        x_pool, x_vlad = outputs
        if vlad:
            outputs = F.normalize(x_vlad, p=2, dim=-1)
        else:
            outputs = F.normalize(x_pool, p=2, dim=-1)
    else:
        outputs = F.normalize(outputs, p=2, dim=-1)
    return outputs


def extract_features_q(model, data_loader, dataset, print_freq=10,
                       vlad=True, pca=None, gpu=None, sync_gather=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    features = []

    if (pca is not None):
        pca.load(gpu=gpu)

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, _, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            outputs = extract_cnn_feature(model, imgs, vlad, gpu=gpu)
            if (pca is not None):
                outputs = pca.infer(outputs)
            outputs = outputs.data.cpu()

            features.append(outputs)

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0 and rank == 0):
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    if (pca is not None):
        del pca

    if (sync_gather):
        # all gather features in parallel
        # cost more GPU memory but less time
        features = torch.cat(features).cuda(gpu)
        all_features = [torch.empty_like(features) for _ in range(world_size)]
        dist.all_gather(all_features, features)
        del features
        all_features = torch.cat(all_features).cpu()[:len(dataset)]
        features_dict = OrderedDict()
        for fname, output in zip(dataset, all_features):
            features_dict[fname[0]] = output
        del all_features
    else:
        # broadcast features in sequence
        # cost more time but less GPU memory
        bc_features = torch.cat(features).cuda(gpu)
        features_dict = OrderedDict()
        for k in range(world_size):
            bc_features.data.copy_(torch.cat(features))
            if (rank == 0):
                print("gathering features from rank no.{}".format(k))
            dist.broadcast(bc_features, k)
            l = bc_features.cpu().size(0)
            for fname, output in zip(dataset[k * l:(k + 1) * l], bc_features.cpu()):
                features_dict[fname[0]] = output
        del bc_features, features

    return features_dict


def extract_features_db(model, data_loader, dataset, list_exist, print_freq=10,
                        features_path=None, features_name=None, vlad=True, pca=None, gpu=None, sync_gather=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    feature_path_db = features_path + features_name + '_rank0.pt'
    if all(list_exist):
        do_nothing=True
    else:
        features = []
        if (pca is not None):
            pca.load(gpu=gpu)

        end = time.time()
        with torch.no_grad():
            for i, (imgs, fnames, _, _, _) in enumerate(data_loader):
                data_time.update(time.time() - end)

                outputs = extract_cnn_feature(model, imgs, vlad, gpu=gpu)
                if (pca is not None):
                    outputs = pca.infer(outputs)
                outputs = outputs.data.cpu()

                features.append(outputs)

                batch_time.update(time.time() - end)
                end = time.time()

                if ((i + 1) % print_freq == 0 and rank == 0):
                    print('Extract Features: [{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Data {:.3f} ({:.3f})\t'
                          .format(i + 1, len(data_loader),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg))

        if (pca is not None):
            del pca
        features = torch.cat(features)
        feature_name = features_path + features_name + '_rank' + str(rank) + '.pt'
        torch.save(features, feature_name)


def extract_filenames_db(data_loader):
    db_filenames = []
    rank = dist.get_rank()
    for i, (imgs, fnames, _, _, _) in enumerate(data_loader):
        db_filenames.append(fnames)
    filename = 'IOVPR/logs/features/db_name_rank' + str(rank) + '.txt'
    with open(filename, 'wb') as fp:
        pickle.dump(db_filenames, fp)

def pairwise_distance_memory(features, features_path, features_name, query=None, gallery=None, max_k=100):
    x = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in query], 0)
    m = x.size(0)
    x = x.view(m, -1).numpy()
    faiss_index = faiss.IndexFlatL2(x.shape[1])
    world_size = dist.get_world_size()
    total_count = 0
    for r in range(world_size):
        y = torch.load(features_path + features_name + '_rank' + str(r) + '.pt')
        n = y.size(0)
        total_count += n
        y = y.view(n, -1).numpy()
        faiss_index.add(y)
    _, sort_idx = faiss_index.search(x, max_k)
    return sort_idx, None, None

def spatial_nms(pred, db_ids, topN):
    assert (len(pred) == len(db_ids))
    pred_select = pred[:topN]
    pred_pids = [db_ids[i] for i in pred_select]
    # find unique
    seen = set()
    seen_add = seen.add
    pred_pids_unique = [i for i, x in enumerate(pred_pids) if not (x in seen or seen_add(x))]
    return [pred_select[i] for i in pred_pids_unique]

def evaluate_all(sort_idx, gt, gallery, recall_topk=[1, 5, 10, 15, 20, 25, 50, 75, 100], nms=False):
    db_fn = [db[0] for db in gallery]
    db_ids = [db[1] for db in gallery]
    if (dist.get_rank() == 0):
        print("===> Start calculating recalls")
    correct_at_n = np.zeros(len(recall_topk))
    viz = []
    for qIx, pred in enumerate(sort_idx):
        if (nms):
            pred = spatial_nms(pred.tolist(), db_ids, max(recall_topk) * 12)

        for i, n in enumerate(recall_topk):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
        viz.append([(db_fn[pred[i]], int(pred[i] in gt[qIx])) for
                    i in range(max(recall_topk))])

    recalls = correct_at_n / len(gt)
    del sort_idx

    if (dist.get_rank() == 0):
        print('Recall Scores:')
        for i, k in enumerate(recall_topk):
            print('  top-{:<4}{:12.1%}'.format(k, recalls[i]))
    return recalls, viz


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model
        self.rank = dist.get_rank()

    def evaluate(self, query_loader, dataset, query, gallery, ground_truth, gallery_loader=None, \
                 features_path = None, features_name = None, \
                 vlad=True, pca=None, rerank=False, gpu=None, sync_gather=False, \
                 nms=False, rr_topk=100, lambda_value=0):
        if (rerank):
            raise NotImplementedError('Re-ranking not possible with FAISS')

        if (gallery_loader is not None):
            print("gallery len:", len(gallery_loader))
            features = extract_features_q(self.model, query_loader, query,
                                          vlad=vlad, pca=pca, gpu=gpu, sync_gather=sync_gather)

            db_files = []
            for i in range(dist.get_world_size()):
                item = features_path + features_name + '_rank' + str(i) + '.pt'
                db_files.append(item)

            list_exist = []
            for file in db_files:
                list_exist.append(os.path.isfile(file))
            if all(list_exist):
                print('all files exist')
            else:
                extract_features_db(self.model, gallery_loader, gallery, list_exist, features_path=features_path,
                                    features_name=features_name, vlad=vlad, pca=pca, gpu=gpu,
                                    sync_gather=sync_gather)

        if self.rank == 0:
            print(psutil.virtual_memory())
            sort_idx, _, _ = pairwise_distance_memory(features, features_path, features_name, query, gallery, max_k=100)
            recalls, viz = evaluate_all(sort_idx, ground_truth, gallery, nms=nms)
            return recalls, viz

        return [], []

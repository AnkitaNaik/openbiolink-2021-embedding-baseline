# -*- coding: utf-8 -*-
#
# eval.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright 2021 OpenBioLink (Modification)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import argparse
import os
import logging
import time
import pickle

import torch.multiprocessing as mp
from dglke.train_pytorch import load_model_from_checkpoint
from dglke.train_pytorch import test, test_mp
from dglke.utils import load_model_config

from openbiolink.obl2021 import OBL2021Dataset, OBL2021Evaluator

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--model_path', type=str, default='ckpts',
                          help='The path of the directory where models are saved.')
        self.add_argument('--batch_size_eval', type=int, default=100,
                          help='The batch size used for evaluation.')
        self.add_argument('--gpu', type=int, default=[-1], nargs='+',
                          help='a list of active gpu ids, e.g. 0')

    def parse_args(self):
        args = super().parse_args()
        return args


def main():
    args = ArgParser().parse_args()

    assert os.path.exists(args.model_path), 'No existing model_path: {}'.format(args.model_path)

    ckpt_path = args.model_path
    config = load_model_config(os.path.join(ckpt_path, 'config.json'))
    args.model_name = config["model"]
    args.hidden_dim = config["emb_size"]
    args.gamma = config["gamma"]
    args.double_ent = config["double_ent"]
    args.double_rel = config["double_rel"]
    args.dataset = config["dataset"]
    args.train = False
    args.valid = False
    args.test = True
    args.strict_rel_part = False
    args.soft_rel_part = False
    args.async_update = False
    args.has_edge_importance = False
    args.mix_cpu_gpu = False
    gpu_id = args.gpu[0]

    ev = OBL2021Evaluator()
    dl = OBL2021Dataset("./data/OBL2021")
    n_entities = dl.num_entities
    n_relations = dl.num_relations

    model = load_model_from_checkpoint(args, n_entities, n_relations, ckpt_path)

    head_neg_score = model.score_func.create_neg(True)
    tail_neg_score = model.score_func.create_neg(False)
    head_neg_prepare = model.score_func.create_neg_prepare(True)
    tail_neg_prepare = model.score_func.create_neg_prepare(False)

    entity_emb = model.entity_emb(torch.arange(dl.num_entities).long(), gpu_id)
    relation_emb = model.relation_emb(torch.arange(dl.num_relations).long(), gpu_id)

    start = time.time()
    n_batches, batches = dl.get_test_batches(args.batch_size_eval)

    top10_tails = []
    top10_heads = []

    for batch in tqdm(batches, total=n_batches):
        
        pos_head_emb = entity_emb[batch[:, 0], :]
        pos_tail_emb = entity_emb[batch[:, 2], :]
        pos_rel = batch[:, 1].long()
        pos_rel_emb = relation_emb[pos_rel, :]

        neg_head, tail = head_neg_prepare(pos_rel, 1, entity_emb, pos_tail_emb, gpu_id, False)
        scores_head = head_neg_score(neg_head, pos_rel_emb, tail,
                                     1, len(batch), dl.num_entities).squeeze(0)
        head, neg_tail = tail_neg_prepare(pos_rel, 1, pos_head_emb, entity_emb, gpu_id, False)
        scores_tail = tail_neg_score(head, pos_rel_emb, neg_tail,
                                     1, len(batch), dl.num_entities).squeeze(0)

        scores_head = dl.filter_scores(
            batch,
            scores_head,
            0,
            float('-Inf')
        )
        scores_tail = dl.filter_scores(
            batch,
            scores_tail,
            2,
            float('-Inf')
        )
        top10_heads.append(torch.topk(scores_head, 10)[1])
        top10_tails.append(torch.topk(scores_tail, 10)[1])

    top10_heads = torch.cat(top10_heads, dim=0)
    top10_tails = torch.cat(top10_tails, dim=0)

    ev.eval(top10_heads, top10_tails, dl.testing)
    print('Evaluation took {:.3f} seconds'.format(time.time() - start))


if __name__ == "__main__":
    main()

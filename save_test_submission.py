import time
import os
import torch
import sys
from tqdm import tqdm
from openbiolink.obl2021 import OBL2021Dataset, OBL2021Evaluator

from dglke.models.infer import ScoreInfer
from dglke.utils import load_model_config


def main():
    model_path = sys.argv[1]

    dl = OBL2021Dataset()
    ev = OBL2021Evaluator()


    config = load_model_config(os.path.join(model_path, 'config.json'))

    model = ScoreInfer(-1, config, model_path)
    model.load_model()
    model = model.model


    head_neg_score = model.score_func.create_neg(True)
    tail_neg_score = model.score_func.create_neg(False)
    head_neg_prepare = model.score_func.create_neg_prepare(True)
    tail_neg_prepare = model.score_func.create_neg_prepare(False)

    entity_emb = model.entity_emb(torch.arange(dl.num_entities).long())
    relation_emb = model.relation_emb(torch.arange(dl.num_relations).long())

    start = time.time()
    n_batches, batches = dl.get_test_batches(100)

    top10_tails = None
    top10_heads = None

    for batch in tqdm(batches, total=n_batches):
        pos_head_emb = entity_emb[batch[:, 0], :]
        pos_tail_emb = entity_emb[batch[:, 2], :]
        pos_rel = batch[:, 1].long()
        pos_rel_emb = relation_emb[pos_rel, :]

        neg_head, tail = head_neg_prepare(pos_rel, 1, entity_emb, pos_tail_emb, -1, False)
        scores_head = head_neg_score(neg_head, pos_rel_emb, tail,
                                     1, len(batch), dl.num_entities).squeeze(0)
        head, neg_tail = tail_neg_prepare(pos_rel, 1, pos_head_emb, entity_emb, -1, False)
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

    res = ev.eval(top10_heads, top10_tails, dl.testing)
    print(res)
    print('Evaluation took {:.3f} seconds'.format(time.time() - start))


if __name__ == "__main__":
    main()
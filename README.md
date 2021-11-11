# openbiolink-2021-embedding-baseline

# Installation

```bash
pip install torch openbiolink dgl==0.4.3
```

Also from folder `dglke-0.1.2` run

```bash
pip install -e .
```

# Training

CLI-commands for each model can be found in `run.sh`. Each command should be run from the project root.

### Hyperparameter

Hyperparameter used to achieve these results:

|          | learning rate | embedding   size | regularization   coefficient | gamma | iterations         |
| -------- | ------------- | ---------------- | ---------------------------- | ----- | ------------------ |
| RESCAL   | 0.05          | 300              | 3.00E-07                     |       | 350000   on 2 GPUs |
| TransR   | 0.1           | 220              | 1.00E-08                     | 12    | 550000   on 2 GPUs |
| ComplEx  | 0.1           | 380              | 2.00E-06                     |       | 360000   on 8 GPUs |
| DistMult | 0.1           | 380              | 4.00E-07                     |       | 950000   on 2 GPUs |
| RotatE   | 0.05          | 128              | 1.00E-07                     | 12    | 550000   on 2 GPUs |
| TransE   | 0.1           | 360              | 3.00E-09                     | 8     | 550000   on 2 GPUs |

# Evaluation

Run from project root:

```bash
python3 save_test_submission.py
  --model_path MODEL_PATH
                        The path of the directory where models are saved.
  --batch_size_eval BATCH_SIZE_EVAL
                        The batch size used for evaluation.
  --gpu GPU             gpu id to be used, e.g. 0, -1 means only cpu is used
```

F.e. for evaluating TransE:

```bash
python3 save_test_submission.py --model_path ./ckpts/DistMult_OBL2021_0 --batch_size_eval 100 --gpu 0
```

# Attribution

This code is based on [dgl-ke](https://github.com/awslabs/dgl-ke)


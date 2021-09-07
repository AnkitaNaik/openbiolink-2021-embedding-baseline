# openbiolink-2021-embedding-baseline

# Installation

```bash
pip install torch openbiolink dgl==0.4.3
```

Also from folder `dglke-0.1.2\python` run

```bash
pip install -e .
```

# Training

CLI-commands for each model can be found in `run.sh`. Each command should be run from the project root.

### Hyperparameter

Hyperparameter used to achieve these results:

|          | learning rate | embedding   size | regularization   coefficient | gamma | iterations         |
| -------- | ------------- | ---------------- | ---------------------------- | ----- | ------------------ |
| ComplEx  | 0.1           | 380              | 2.00E-06                     |       | 360000   on 8 GPUs |
| DistMult | 0.1           | 380              | 4.00E-07                     |       | 950000   on 2 GPUs |
| RotatE   | 0.05          | 128              | 1.00E-07                     | 12    | 550000   on 2 GPUs |
| TransE   | 0.1           | 360              | 3.00E-09                     | 8     | 550000   on 2 GPUs |

# Evaluation

Run from project root:

```bash
python3 save_test_submission.py {path to model}
```

where `path to model` is  the path to the folder containing the model that you want to evaluate.  F.e. if it is the first TransE Model `{path to model}` would be `ckpts/TransE_l2_OBL_0`.

# Attribution

This code is based on [dgl-ke](https://github.com/awslabs/dgl-ke)


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

# Evaluation

Run from project root:

```bash
python3 save_test_submission.py {path to model}
```

where `path to model` is  the path to the folder containing the model that you want to evaluate.  F.e. if it is the first TransE Model `{path to model}` would be `ckpts/TransE_l2_OBL_0`.

# Attribution

This code is based on [dgl-ke](https://github.com/awslabs/dgl-ke)


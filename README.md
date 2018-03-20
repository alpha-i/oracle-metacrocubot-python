#oracle metacrocubot

### Create conda environment
```bash
$ conda create -n metacrocubot python=3.5 numpy
$ source activate metacrocubot
```

### Install dependencies

```bash
$ pip install -U setuptools --ignore-installed --no-cache-dir
$ pip install -r dev-requirements.txt --src $CONDA_PREFIX
```

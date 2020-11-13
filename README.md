# language-models-are-knowledge-graphs-pytorch
Language models are open knowledge graphs ( work in progress )

A non official reimplementation of [Language models are open knowledge graphs](https://arxiv.org/abs/2010.11967)

The implemtation of Match is in process.py

Map function is still in progress


### Execute Match section

Do note the extracted results is still quite noisy and should then filtered based on relation unique pair frequency 

```
python extract.py examples/bob_dylan.txt bert-large-cased-bob_dynlan.jsonl --language_model bert-large-cased --use_cuda true
```

### Environment setup


This repo is run using virtualenv 

```
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
```
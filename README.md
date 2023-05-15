# barkify
Barkify: an unoffical repo for training bark-like generative model

## Process dataset
We do our experiment on LJspeech. Follow the instrcutions in `process.ipynb`. <br>
For chinese, we test a famous steamer named Fengge. It shows an acceptable result but worse than our other TTS repo.

## Training
Stage1 stands for text to semantic and stage2 stands for semantic to acoustic. <br>
You should config paramters in the `configs/barkify.yaml`. We use one A100 to train our model (both S1&S2). 
```
# training stage 1 or 2
python trainer.py start_path=/path/to/your/work_env stage=1 name=<dataset>
python trainer.py start_path=/path/to/your/work_env stage=2 name=<dataset>
```

## Inference
Directly use `infer.ipynb` and follow the instrcutions to infer your model.

## Roadmap
- [x] Construct a basic training code for bark-like generative model
- [x] Test one speaker scenario
- [ ] Test multi speaker scenario
- [ ] Test speaker semantic prompting
- [ ] Test speech/audio acoustic prompting
- [ ] Test variable length data(as we use a fixed length now)
- [ ] Long-form generation
- [ ] Support more language

## Appreciation
- [bark](https://github.com/suno-ai/bark/) is a transformer-based text-to-audio model.
- [Vall-E](https://github.com/lifeiteng/vall-e) is an unofficial PyTorch implementation of VALL-E.

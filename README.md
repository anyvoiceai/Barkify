# barkify
Barkify: an unoffical repo for training Bark, a text-prompted generative audio model by suno-ai. 

Bark has two GPT style models which is compatible for prompting and other tricks from NLP. Bark realize a great real world tts result but the repo itself doesn't a training recipe. We want to conduct some experiments or train this model. Here we release our basic training code which might be a guidance of training for open source community. 

## Process dataset
We do our experiment on LJspeech. Follow the instrcutions in `process.ipynb`. <br>
For Chinese, we test a famous steamer named `峰哥亡命天涯`. It shows an acceptable result but worse than our other TTS repo.
For English, we test LibriTTS dataset. It works fine and basic items in our roadmap have been proved.

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
We have already achieve the following items and we will release our code soon.
- [x] Construct a basic training code for bark-like generative model
- [x] Test one speaker scenario
- [x] Test multi speaker scenario
- [x] Test speaker semantic prompting
- [x] Test speech/audio acoustic prompting
- [x] Test variable length data(as we use a fixed length now)

These items are pretty data-hungry or rely on massive GPUs. <br>
So we are open to any sponsors or collaborators to finish these jobs. <br>
You could contact us by QQ: 3284494602 or email us at 3284494602@qq.com

- [ ] Long-form generation(which may be longer than 1min.)
- [ ] Support more language(especially for ZH)
- [ ] Paralanguage modeling in the text input
- [ ] Speaker generation by text prompts
- [ ] Emotion/Timbre/Rhythm controlling by text/acoustic prompts
- [ ] Add/Remove background noise(which might be important for downstream tasks) 

## Appreciation
- [bark](https://github.com/suno-ai/bark/) is a transformer-based text-to-audio model.
- [Vall-E](https://github.com/lifeiteng/vall-e) is an unofficial PyTorch implementation of VALL-E.

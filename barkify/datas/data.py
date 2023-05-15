import json
from os.path import join as pjoin

import random
import torch
import numpy as np
from tqdm import tqdm

from .tokenizer import SplitTokenizer

class Dataset:
    def __init__(self, start_path, file, 
                tokenizer: SplitTokenizer = None, add_prompt = False
            ):

        self.start_path = start_path
        self.tokenizer = tokenizer
        self.add_prompt = add_prompt # add prompt for semantic or acoustic

        with open(pjoin(start_path, 'meta', file+'.json')) as f:
            self._datas = [json.loads(i) for i in f.readlines()]
            g2p_already = any([data.get('g2p', False) for data in self._datas])
        
        if not g2p_already:
            print("g2p all texts.")
            for data in tqdm(self._datas):
                if not data.get('g2p', False):
                    data['g2p'] = self.tokenizer.g2p(data['text'])

            with open(pjoin(start_path, 'meta', file+'.json'), "w") as f:
                for data in self._datas: 
                    f.writelines(json.dumps(data, ensure_ascii=False)+'\n')

    def __getitem__(self, idx):
        batch = {}
        data = self._datas[idx]

        batch['name'] = data['name']
        batch['text'] = self.tokenizer.token2id(data['g2p'])

        semantic_path = pjoin(self.start_path, 'semantic_idx', data['name'])
        codec_path = pjoin(self.start_path, 'encodec_idx', data['name'])
        batch['semantic'] = torch.from_numpy(np.load(semantic_path))
        batch['encodec'] = torch.from_numpy(np.load(codec_path))

        if self.add_prompt:
            raise NotImplementedError

        return batch

    def __len__(self):
        return len(self._datas)

def Text2semanticCollateFn(
        batches, 
        text_window=512, semantic_window=512, # set training window size.
        text_token_num=210, fixed_length=True, ign_idx=-100, **kwargs
    ):
    '''
    logits not following bark: text(210) + pad_text(1) + eos(1) + infer(1) + semantic(2048)
    returns:
        input: [text with pad(512), infer(1), semantic with pad(512), eos(1)]
        tgt: [ign_idx(512), ign_idx(1), (semantic, ign_idx)(512), eos(1)]
    '''

    text_pad_token, semantic_pad_token = text_token_num, text_token_num + 1
    infer_token, text_offset = text_token_num + 2, text_token_num + 3

    semantic, text = [], []
    semantic_length, text_length = [], []
    names = []
    
    for batch in batches:
        names.append(batch['name'])

        _text = np.array(batch['text'])
        if fixed_length:
            _text = _text[-text_window:] # TODO: drop long text?
        text_length.append(len(_text))
        
        _text = np.pad(
                _text, (0, text_window - len(_text)),
                constant_values = text_pad_token
            )

        _semantic = batch['semantic'] + text_offset # different from bark, we add offset to semantic tokens rather than text.
        if fixed_length:
            _semantic = _semantic[-semantic_window:]
        semantic_length.append(len(_semantic))

        _semantic = np.pad(
                _semantic, (0, semantic_window - len(_semantic)),
                constant_values = semantic_pad_token
            )
        
        semantic.append(_semantic), text.append(_text)
    
    text, semantic = torch.from_numpy(np.stack(text, axis=0)), torch.from_numpy(np.stack(semantic, axis=0)) 
    text_length, semantic_length = torch.tensor(text_length), torch.tensor(semantic_length)
    
    B = text.shape[0]
    inputs = torch.cat([text, torch.ones(B, 1) * infer_token, semantic, torch.ones(B, 1) * semantic_pad_token], -1)
    tgt = torch.cat([torch.ones(B, text_window + 1) * ign_idx, semantic, torch.ones(B, 1) * semantic_pad_token], -1)
    tgt[:, text_window+1:][(torch.arange(0, semantic_window + 1) > semantic_length[:, None])] = -100

    batch = dict(names = names, 
        input_ids = inputs.long(), labels = tgt.long(),
        semantic_length = semantic_length, text_length = text_length
    )
    return batch

def Semantic2coarseCollateFn(
        batches, Q_size, 
        semantic_window=256, semantic_to_coarse_ratio=3, 
        semantic_token_num=2048, coarse_num=1024, ign_idx=-100,
        slice_range = 60, # window size in bark
        **kwargs
    ):
    '''
    logits following bark: semantic(2048) + pad(1) + infer(1) + coarse(1024*Q)
    returns:
        input: [semantic with pad(256), infer(1), coarse with pad(768)]
        tgt: [ign_idx(256), ign_idx(1), (coarse, ign_idx)(768)]
    '''

    semantic_pad_token = coarse_pad_token = semantic_token_num
    infer_token = semantic_token_num + 1

    semantic_window = semantic_window // 2 * 2
    acoustic_pad = int(np.ceil((semantic_window * semantic_to_coarse_ratio) / Q_size) * Q_size)
    semantic, coarse = [], []
    semantic_length, coarse_length = [], []
    names = []
    
    for batch in batches:
        names.append(batch['name'])
        semantic_start = random.randint(0, np.max([0, len(batch['semantic']) - (semantic_window - slice_range)]))
        semantic_start = (semantic_start // Q_size) * Q_size # add
        _semantic = batch['semantic'][semantic_start:semantic_start+semantic_window]
        semantic_length.append(len(_semantic))
        
        _semantic = np.pad(
                _semantic, (0, semantic_window - len(_semantic)),
                constant_values = semantic_pad_token
            )
        
        coarse_start = semantic_start * semantic_to_coarse_ratio
        coarse_end = coarse_start + acoustic_pad
                    
        _coarse = batch['encodec'][:Q_size] + 2 + semantic_pad_token # add pad and infer token
        for i in range(1, _coarse.shape[0]):
            _coarse[i:] += coarse_num
        _coarse = _coarse.T.reshape(-1)[int(coarse_start) : int(coarse_end)]
        coarse_length.append(len(_coarse))

        _coarse = np.pad(
                _coarse, (0, acoustic_pad - len(_coarse)),
                constant_values = coarse_pad_token
                )
        
        semantic.append(_semantic), coarse.append(_coarse)
        
    semantic, coarse = torch.from_numpy(np.stack(semantic, axis=0)), torch.from_numpy(np.stack(coarse, axis=0)) 
    semantic_length, coarse_length = torch.tensor(semantic_length), torch.tensor(coarse_length)
    
    inputs = torch.cat([semantic, torch.ones(semantic.shape[0], 1) * infer_token, coarse], -1)
    tgt = torch.cat([torch.ones(semantic.shape[0], semantic_window + 1) * ign_idx, coarse], -1)
    tgt[(tgt == semantic_pad_token)|(tgt == coarse_pad_token)] = ign_idx

    batch = dict(names = names, 
        input_ids = inputs.long(), labels = tgt.long(),
        semantic_length = semantic_length, coarse_length = coarse_length
    )
    return batch

if __name__ == "__main__":
    from barkify.datas.tokenizer import ZHTokenizer
    dataset = Dataset('/root/intern/bark/barkify/work_env', 'train', ZHTokenizer())
    print(dataset[0]['text'])
# TODO: integrate with NeMo tokenizer?
import string
from pypinyin import lazy_pinyin, Style
from g2p_en import G2p

class SplitTokenizer:
    def __init__(self, **kwargs):
        self._token2id = {i:idx + 1 for idx, i in enumerate(ENGLISH_PRONUNCIATION_LIST)} 
        self._id2token = {idx + 1:i for idx, i in enumerate(ENGLISH_PRONUNCIATION_LIST)} 
    
    def __call__(self, text):
        text = self.g2p(text)
        return self.token2id(text)
    
    def g2p(self, text):
        # text: I'm Tom. -> [i, ', m, <space>, t, o, m, .]
        text = text.lower()
        return " ".join(["<space>" if i == ' ' else i for i in list(text)])
    
    def token2id(self, text):
        token = []
        for i in text.split():
            if i == ' ':
                token.append(self._token2id['<space>'])
            elif self._token2id.get(i, None):
                token.append(self._token2id[i])

        return token

class PhonemeTokenizer(SplitTokenizer):
    def __init__(self, **kwargs):
        self._g2p = G2p()
        self._token2id = {i:idx + 1 for idx, i in enumerate(self._g2p.phonemes)} 
        self._id2token = {idx + 1:i for idx, i in enumerate(self._g2p.phonemes)} 

    def g2p(self, text):
        text = self._g2p(text)
        return " ".join(["<space>" if i == ' ' else i for i in text])

class ZHTokenizer(SplitTokenizer):
    def __init__(self, **kwargs):
        # A basic english and chinese G2P tokenizer 
        self._token2id = {i:idx + 1 for idx, i in enumerate(PINYIN_PRONUNCIATION_LIST)} 
        self._id2token = {idx + 1:i for idx, i in enumerate(PINYIN_PRONUNCIATION_LIST)} 
    
    def g2p(self, text):
        text = text.lower()
        initials = lazy_pinyin(text, neutral_tone_with_five=False, style=Style.INITIALS, strict=False)
        finals = lazy_pinyin(text, neutral_tone_with_five=False, style=Style.FINALS_TONE3)
        
        text_phone = []
        for _o in zip(initials, finals):
            if _o[0] != _o[1] and _o[0] != '':
                _o = ['@'+i for i in _o]
                text_phone.extend(_o)
            elif _o[0] != _o[1] and _o[0] == '':
                text_phone.append('@'+_o[1])
            else:
                text_phone.extend(["<space>" if i == ' ' else i for i in list(_o[0])])

        return " ".join(text_phone)

ENGLISH_PRONUNCIATION_LIST = list(string.ascii_lowercase) + list(",.?!") + ['<space>']
PINYIN_PRONUNCIATION_LIST = ENGLISH_PRONUNCIATION_LIST + [
# PINYIN_PRONUNCIATION_LIST = [
    '@w', '@uo3', '@y', '@i3', '@j', '@ing1', '@b', '@a3', '@o1', '@l', '@a1', '@h', '@ei1', '@e', 
    '@n', '@iou4', '@sh', '@i4', '@ie2', '@z', '@ai4', '@a4', '@m', '@g', '@ou3', '@t', '@uo2', '@i', '@q', 
    '@vn2', '@uo1', '@zh', '@i1', '@d', '@ao4', '@uei4', '@a2', '@a', '@e4', '@ing3', '@ei', '@u4', '@uan2', 
    '@f', '@an4', '@en2', '@c', '@uo4', '@uei1', '@iou1', '@ei2', '@e2', '@r', '@en4', '@eng1', '@e1', '@en1',
    '@ou4', '@ang4', '@p', '@eng2', '@ong4', '@u2', '@iang2', '@van1', '@ian1', '@ei4', '@er3', '@ia1', '@ou2', 
    '@ao3', '@ou1', '@er2', '@s', '@i2', '@v4', '@x', '@ian4', '@ong1', '@uan3', '@uang2', '@ing4', '@ch', '@vn3',
    '@uen1', '@ai1', '@an3', '@eng4', '@ing2', '@ve4', '@k', '@ang3', '@en3', '@ai2', '@ian3', '@er4', '@ai3', 
    '@uai4', '@ian2', '@ao1', '@eng3', '@ia4', '@n2', '@ang1', '@ie3', '@uen3', '@iou3', '@ei3', '@in4', '@v3', 
    '@uen4', '@an2', '@iang1', '@in1', '@u3', '@ve2', '@e3', '@iang4', '@ia', '@an1', '@in3', '@iao4', '@ang2', 
    '@vn1', '@iao3', '@u1', '@ie1', '@ie4', '@v2', '@uei2', '@iong1', '@iao1', '@o2', '@uei3', '@in2', '@iong4',
    '@ve1', '@uang1', '@iang3', '@uan4', '@iou2', '@en', '@uan1', '@ia2', '@ua1', '@ong3', '@van4', '@van2', '@uang3', 
    '@iao2', '@ua4', '@ong2', '@uen2', '@iong3', '@er', '@v1', '@uang4', '@ia3', '@ve3', '@ua2', '@van3', '@ao2',
    '@o4', '@ua3', '@vn4', '@iong2', '@io1', '@uai1', '@ou', '@uai2', '@ua', '@ueng1', '@o', '@uai3', '@o3', '@uo',
] + list("、，。")
# ] + list(string.ascii_lowercase) + list("、，。")

if __name__ == "__main__":
    tokenizer = ZHTokenizer()
    print(tokenizer.g2p("I'm tom."))
    print(tokenizer("I'm tom."))

    tokenizer = PhonemeTokenizer()
    print(tokenizer.g2p("I'm tom."))
    print(tokenizer("I'm tom."))

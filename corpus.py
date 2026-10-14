import argparse
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel,AutoTokenizer

MODEL='sentence-transformers/all-MiniLM-L6-v2'
REVISION='1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def load_archive(path):
    with tarfile.open(path) as archive:
        return [[json.loads(line) for line in archive.extractfile(f'data/{name}.jsonl')]
                for name in ['corpus','claims_train','claims_dev']]


def embed(texts,tokenizer,model,device,progress=True):
    values=[];truncated=0
    with torch.inference_mode():
        for offset in range(0,len(texts),32):
            strings=texts[offset:offset+32]
            lengths=[len(tokenizer(s,truncation=False)['input_ids']) for s in strings]
            truncated+=sum(n>256 for n in lengths)
            inputs=tokenizer(strings,padding=True,truncation=True,max_length=256,return_tensors='pt')
            inputs={k:v.to(device) for k,v in inputs.items()};hidden=model(**inputs).last_hidden_state
            mask=inputs['attention_mask'].unsqueeze(-1)
            values.extend(torch.nn.functional.normalize((hidden*mask).sum(1)/mask.sum(1),dim=1).cpu().numpy())
            if progress and offset%1024==0:print('encoded',offset,'/',len(texts),flush=True)
    return np.array(values,dtype='float32'),truncated


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path('data/scifact.tar.gz'))
    parser.add_argument('--device',default='mps' if torch.backends.mps.is_available() else 'cpu');args=parser.parse_args()
    documents,train,dev=load_archive(args.data)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModel.from_pretrained(MODEL,revision=REVISION).to(args.device).eval()
    texts=[d['title']+' '+' '.join(d['abstract']) for d in documents]+[c['claim'] for c in train+dev]
    vectors,truncated=embed(texts,tokenizer,model,args.device)
    np.savez_compressed('data/embeddings.npz',documents=vectors[:len(documents)],
        train=vectors[len(documents):len(documents)+len(train)],dev=vectors[len(documents)+len(train):])
    Path('data/embedding-manifest.json').write_text(json.dumps({'encoder':MODEL,'revision':REVISION,'max_tokens':256,
        'pooling':'L2-normalized mean of attention-masked final states, including special tokens (model-card recipe)',
        'texts':len(texts),'truncated_texts':truncated,'data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'embedding_sha256':hashlib.sha256(Path('data/embeddings.npz').read_bytes()).hexdigest()},indent=2)+'\n')


if __name__=='__main__':main()

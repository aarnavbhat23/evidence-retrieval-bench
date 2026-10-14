import hashlib
import urllib.request
from pathlib import Path
import truststore

URL='https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz'
SHA256='11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be'


if __name__=='__main__':
    truststore.inject_into_ssl();data=urllib.request.urlopen(URL,timeout=120).read()
    if hashlib.sha256(data).hexdigest()!=SHA256:raise ValueError('SciFact release changed; inspect before updating the checksum')
    Path('data').mkdir(exist_ok=True);Path('data/scifact.tar.gz').write_bytes(data);print('Verified SciFact archive')

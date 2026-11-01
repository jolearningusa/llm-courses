"""Embed offline presenter controls in both language folders; safe to run again."""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent

def build():
    js = (HERE / 'presenter_mode.js').read_text()
    block = '<script id="course-presenter">' + js + '</script>'
    paths = sorted(HERE.glob('*.html')) + sorted((HERE / 'zh').glob('*.html'))
    for path in paths:
        text = path.read_text()
        text = re.sub(r'<script id="course-presenter">.*?</script>', '', text, flags=re.S)
        if '</body>' not in text:
            raise ValueError('missing HTML body: ' + str(path))
        path.write_text(text.replace('</body>', block + '</body>'))
    print('Embedded presenter controls in', len(paths), 'decks')

if __name__ == '__main__':
    build()

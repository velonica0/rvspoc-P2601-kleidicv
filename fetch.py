import urllib.request
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = []
    def handle_data(self, d):
        self.text.append(d)
    def get_data(self):
        return ''.join(self.text)

try:
    req = urllib.request.Request('https://rvspoc.org/P2601/', headers={'User-Agent': 'Mozilla/5.0'})
    html = urllib.request.urlopen(req).read().decode('utf-8')
    s = MLStripper()
    s.feed(html)
    print(s.get_data())
except Exception as e:
    print("Error:", e)

from http.cookiejar import debug
from sys import flags

from flask import Flask, request, render_template_string
import os
import joblib

app = Flask(__name__)

pklfile = os.path.dirname(__file__) +"/lang/freq.pkl"
clf = joblib.load(pklfile)

@app.route('/', methods =['GET','POST'])
def index():
    text = request.form.get('text', '')
    msg =''
    if text:
        lang = detect_lang(text)
        msg = "판정결과" + lang
return render_template_string('''
    <html>
    <body>
        <form method = "post">
        <textarea name ="text" rows = "8" cols="40">{{text}}</textarea>
        <p><input type="submit" value="판정" </p>
        <p>{{msg}}</p>
        </form>
    <body>
    <html>
''', text=text,msg=msg)

def detect_lang(text):
    text = text.lower()
    code_a, code_z = (ord("a"), ord("Z"))
    cnt = [0 for i in range(26)]
    for ch in text:
        n = ord(ch) - code_a
        if 0 <= n < 26: cnt[n] +=1
    total = sum(cnt)
    if total == 0: return
    freq = list(map(lambda n:n / total,cnt))

    res = clf.predict([freq])
    lang_dic = {'en' : '영어', "fr": "프랑스어". "id": "인도네시아어",
                "tl", "타갈로그어"}
    return lang_dic.get(res[0], '알수없는 언어')
if__name__ == "__main__":
    app.run(debug=True)
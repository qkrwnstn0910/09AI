from PIL import Image
import numpy as np
import os,re

search_dir = "./caltech101/101_ObjectCategories"
cache_dir= "./caltech101/cache_avhash"
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

def average_hash(fname,size=16):
    print(f"average_hash() 호출됨 - 처리할 파일: {fname}")
    fname2 = fname[len(search_dir):]
    fname2 = fname2.replace("\\","/")
    cache_file = cache_dir + "/" + fname2.replace('/','_')+"csv"

    if not os.path.exists(cache_file):
        img = Image.open(fname)
        img = img.convert('L').resize((size, size), Image.Resampling.LANCZOS)
        pixels = np.array(img.getdata()).reshape((size,size))
        avg = pixels.mean()
        px = 1*(pixels > avg)
        np.savetxt(cache_file,px, fmt="%.0f",delimiter=",")
    else:
        px = np.loadtxt(cache_file,delimiter=",")
    return px
def hamming_dist(a,b):
    aa = a.reshape(1,-1)
    ab = b.reshape(1,-1)
    dist = (aa != ab).sum()
    return dist

def enum_all_files(path):
    for root, dirs, files in os.walk(path):
        for f in files:
            fname = os.path.join(root,f)
            if re.search(r'\.(jpg|jpeg|png)$',fname):
                yield fname

def find_image(fname,rate):
    src = average_hash(fname)
    for fname in enum_all_files(search_dir):
        fname2 = fname.replace("\\","/")
        dst = average_hash(fname2)
        diff_r = hamming_dist(src,dst) / 256
        if diff_r < rate:
            yield (diff_r, fname2)
# 검색할 기준 이미지 경로 설정
srcfile = search_dir + "/chair/image_0016.jpg"
html = ""
sim = list(find_image(srcfile, 0.25))
sim = sorted(sim, key=lambda x:x[0])
for r, f in sim:
    print(r, ">", f)
    f2 = "." + f
    s = '<div style="float:left;"><h3>[ 차이 :' + str(r) + '-' + \
        os.path.basename(f) + ']</h3>'+ \
        '<p><a href="' + f2 + '"><img src="' + f2 + '">'+ \
        '</a></p></div>'
    html += s

# HTML로 출력하기
html = """<html><head><meta charset="utf8"></head>
<body><h3>원래 이미지</h3><p>
<img src='{0}'></p>{1}
</body></html>""".format("."+srcfile, html)
with open("./saveFiles/avhash-search-output.html", "w", encoding="utf-8") as f:
    f.write(html)
print("HTML 저장 Ok")


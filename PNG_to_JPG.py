from PIL import Image

import os
import sys

if len(sys.argv) != 3:
    print("Usage: PNG_to_JPG.py [source dir] [destination dir] ")
    sys.exit()


source = sys.argv[1]
destination = sys.argv[2]

if not (source.endswith("/") or source.endswith("\\")):
    source = source + "/"
    
if not (destination.endswith("/") or destination.endswith("\\")):
    destination = destination + "/"

files = os.listdir(source)
i = 0
for file in files:
    # Ignore .DS_Store for those on Mac
    if i == ".DS_Store":
        continue
    
    if file.endswith(".png"):
        im1 = Image.open(f"{source}/{file}")
        path = destination + str(i) + ".jpg"
        print(path)
        im1.save(path)
        i+=1
    else:
        continue

print("Done")

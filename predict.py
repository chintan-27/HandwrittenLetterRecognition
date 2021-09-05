import os

dir = 'subimage'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))

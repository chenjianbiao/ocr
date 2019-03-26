from PIL import Image
import matplotlib.pyplot as plt
img=Image.open("./img/aa.jpg")

d=list(img.getdata())
print(d.width)
print(len(d))
print(d[106400-1])

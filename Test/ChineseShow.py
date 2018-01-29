from matplotlib.font_manager import fontManager

import matplotlib.pyplot as plt
import os

# from matplotlib.font_manager import _rebuild
# _rebuild()

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(111)
plt.subplots_adjust(0, 0, 1, 1, 0, 0)
plt.xticks([])
plt.yticks([])

x, y = 0.05, 0.08
fonts = [font.name for font in fontManager.ttflist if
         os.path.exists(font.fname) and os.stat(font.fname).st_size>1e6]
font = set(fonts)
dy = (1.0-y)/( len(fonts)/4 + (len(fonts)%4!=0) )

for font in fonts:
    # font == 'Microsoft YaHei'
    t = ax.text(x,y, u'子节点阿斯顿发送到发送到', {'fontname':font, 'fontsize':14}, transform=ax.transAxes)
    ax.text(x, y-dy/2, font, transform=ax.transAxes)
    x += 0.25
    if x >= 1.0:
        y += dy
        x = 0.05

print(fonts)
plt.show()



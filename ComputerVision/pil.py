from PIL import Image

# read an grayscale image
pil_im = Image.open('image/images.jpeg').convert('L')

# thumbnails
pil_im.thumbnail((128, 128))

# copy a region
box = (100, 100, 200, 200)
region = pil_im.crop(box)

# rotate an image at 180 degree
region = region.transpose(Image.ROTATE_180)

# paste the region
pil_im.paste(region, box)

# resize
out = pil_im.resize((128, 128))

# rotate
out = pil_im.rotate(45)

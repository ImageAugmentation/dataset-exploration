# Make transparent
def make_transparent(pil_image, color_to_be_transparent=(0, 0, 0)):
    img = pil_image.copy().convert("RGBA")
    datas = img.getdata()

    transparent_image = []
    for p in datas:
        if p[0:3] == color_to_be_transparent:
            transparent_image.append((*p[0:3], 0))
        else:
            transparent_image.append(p)
    img.putdata(transparent_image)
    return img


def convert_2d_mask_to_3d(mask_2d):
    return mask_2d.reshape((mask_2d.shape[0], mask_2d.shape[1], 1))

from PIL import Image, ImageEnhance
import numpy as np
import torchvision.transforms as transforms


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


def adjust_color(image_tensor, mask_tensor, label, factor):
    image = transforms.ToPILImage()(image_tensor)
    image_np = np.asarray(image)
    mask_semantic_np = np.asarray(transforms.ToPILImage()(mask_tensor))
    mask_semantic_3d = convert_2d_mask_to_3d(mask_semantic_np)

    # Create mask for car
    mask_car = np.where(mask_semantic_3d != label, image_np * 0, np.ones_like(image_np) * 255)

    # Cut masked area
    masked_cars = Image.fromarray(np.zeros_like(image_np))
    masked_cars.paste(image, (0, 0), Image.fromarray(mask_car).convert('L'))

    # Make transparent
    transparent_mask = make_transparent(masked_cars, (0, 0, 0))

    # Change color of cars
    enhancer = ImageEnhance.Color(transparent_mask)
    colored_cars = enhancer.enhance(factor)
    colored_background = image.copy()
    colored_background.paste(colored_cars, (0, 0), colored_cars)

    return transforms.ToTensor()(colored_background)

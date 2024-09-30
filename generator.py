import os
from PIL import Image, ImageDraw, ImageFont
import random
import math
import matplotlib.pyplot as plt


#Loads grassy images for bacgrkound
grass_images_dir = "grass_images"
background_images = [Image.open(os.path.join(grass_images_dir, img)) for img in os.listdir(grass_images_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]


#Creates YOLO dataset split (train and val)
output_dir = "yolo_dataset"
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "val")
images_dir = os.path.join(train_dir, "images")
labels_dir = os.path.join(train_dir, "labels")
test_images_dir = os.path.join(test_dir, "images")
test_labels_dir = os.path.join(test_dir, "labels")

for dir_path in [images_dir, labels_dir, test_images_dir, test_labels_dir]:
    os.makedirs(dir_path, exist_ok=True)



# Draw similar shapes to original data
def draw_shape(draw, shape_type, bounds, color):
    x0, y0, x1, y1 = bounds

    if shape_type == 'circle':
        draw.ellipse([x0, y0, x1, y1], fill=color)
    elif shape_type == 'semicircle':
        draw.pieslice([x0, y0, x1, y1], start=180, end=360, fill=color)
    elif shape_type == 'quarter_circle':
        draw.pieslice([x0, y0, x1, y1], start=270, end=360, fill=color)
    elif shape_type == 'triangle':
        draw.polygon([(x0 + (x1 - x0) / 2, y0), (x1, y1), (x0, y1)], fill=color)
    elif shape_type == 'rectangle':
        rect_width = (x1 - x0)
        rect_height = rect_width * 0.5
        x1 = x0 + rect_width
        y1 = y0 + rect_height
        draw.rectangle([x0, y0, x1, y1], fill=color)
    elif shape_type == 'pentagon':
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        size = (x1 - x0) / 2
        points = [(cx + size * math.cos(2 * math.pi * i / 5 - math.pi / 2),
                   cy + size * math.sin(2 * math.pi * i / 5 - math.pi / 2)) for i in range(5)]
        draw.polygon(points, fill=color)
    elif shape_type == 'star':
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        size = (x1 - x0) / 2
        points = []
        for i in range(10):
            r = size if i % 2 == 0 else size / 2
            angle = i * (math.pi / 5) - math.pi / 2
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
        draw.polygon(points, fill=color)
    elif shape_type == 'cross':
        #The cross is really a plus
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2  
        arm_width = (x1 - x0) / 5  
        arm_length = (x1 - x0) / 2 
        draw.rectangle([cx - arm_width / 2, cy - arm_length, cx + arm_width / 2, cy + arm_length], fill=color)
        draw.rectangle([cx - arm_length, cy - arm_width / 2, cx + arm_length, cy + arm_width / 2], fill=color)



#Makes sure bboxes are not oustide image
def is_bbox_valid(bbox, image_size):

    bbox_left, bbox_top, bbox_right, bbox_bottom = bbox

    if bbox_left < 0 or bbox_top < 0 or bbox_right > image_size[0] or bbox_bottom > image_size[1]:
        return False
    return True


# Shape to ID (for labeling)
shape_classes = {
    'circle': 0,
    'semicircle': 1,
    'quarter_circle': 2,
    'triangle': 3,
    'rectangle': 4,
    'pentagon': 5,
    'star': 6,
    'cross': 7
}

# Letter to ID (also for labeling)
letter_classes = {chr(i + 65): i + 8 for i in range(26)}

# The shape generator
def generate_shape_images(num_images_per_shape=10, image_size=(416, 416)):
    images_and_labels = []
    for shape_type in shape_classes.keys():
        # Triangle and Rectangle needs better representation in dataset distribution (they keep getting filtered out bc of bbox)
        if shape_type in ['triangle', 'rectangle']:
            num_images = num_images_per_shape * 6
        else:
            num_images = num_images_per_shape
        for i in range(num_images):

            background = random.choice(background_images).resize(image_size)
            image = background.copy()

            shape_color = random.choice(['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'white', 'black', 'magenta', 'gray', (216, 191, 216), (255, 255, 224)])

            padding = 10
            max_shape_size = min(image_size) - 2 * padding

            shape_size = random.randint(int(max_shape_size * 0.8), max_shape_size)
            x0 = random.randint(padding, image_size[0] - shape_size - padding)
            y0 = random.randint(padding, image_size[1] - shape_size - padding)
            x1 = x0 + shape_size
            y1 = y0 + shape_size

            # Draws the shape into a Image
            shape_layer = Image.new('RGBA', (shape_size, shape_size), (0, 0, 0, 0))
            shape_draw = ImageDraw.Draw(shape_layer)
            draw_shape(shape_draw, shape_type, (0, 0, shape_size, shape_size), shape_color)

            shape_mask = shape_layer.split()[-1]
            shape_bbox = shape_mask.getbbox()

            # Skips bad images
            if shape_bbox is None:
                print(f"Skipping image {i + 1} due to empty shape bounding box.")
                continue

            # Rotate!
            rotation_angle = random.randint(0, 360)
            rotated_shape = shape_layer.rotate(rotation_angle, expand=True)

            # Gets bbox
            rotated_shape_bbox = rotated_shape.getbbox()

            # Skips bad image again
            if rotated_shape_bbox is None:
                print(f"Skipping image {i + 1} due to empty bounding box after rotation.")
                continue

            rotated_width, rotated_height = rotated_shape.size
            paste_x = x0 - (rotated_width - shape_size) // 2
            paste_y = y0 - (rotated_height - shape_size) // 2

            image.paste(rotated_shape, (paste_x, paste_y), rotated_shape)

            rotated_shape_mask = shape_mask.rotate(rotation_angle, expand=True)
            shape_bbox = rotated_shape_mask.getbbox()

            shape_bbox_left = paste_x + shape_bbox[0]
            shape_bbox_top = paste_y + shape_bbox[1]
            shape_bbox_right = paste_x + shape_bbox[2]
            shape_bbox_bottom = paste_y + shape_bbox[3]

            # Important!! Checks if bbox is out of image, then skips image generation (prevents corrupt images)
            if not is_bbox_valid((shape_bbox_left, shape_bbox_top, shape_bbox_right, shape_bbox_bottom), image_size):
                print(f"Skipping image {i + 1} due to out of bounds bounding box")
                continue

            bbox_center_x = (shape_bbox_left + shape_bbox_right) / 2 / image_size[0]
            bbox_center_y = (shape_bbox_top + shape_bbox_bottom) / 2 / image_size[1]
            bbox_width = (shape_bbox_right - shape_bbox_left) / image_size[0]
            bbox_height = (shape_bbox_bottom - shape_bbox_top) / image_size[1]

            shape_class_id = shape_classes[shape_type]
            shape_annotation = f"{shape_class_id} {bbox_center_x:.6f} {bbox_center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"

            images_and_labels.append((image, shape_annotation))

            print(f"Shape Image {i + 1} generated!")

    return images_and_labels

# The character generator
def generate_character_images(num_images_per_char=10, image_size=(416, 416)):
    images_and_labels = []
    for letter in letter_classes.keys():
        for i in range(num_images_per_char):

            background = random.choice(background_images).resize(image_size)
            image = background.copy()

            letter = random.choice([chr(i) for i in range(65, 91)])  # A-Z
            font_size = int(image_size[1] * 0.6)  # Make font size large (80% of image height)
            font = ImageFont.truetype("arialbd.ttf", size=font_size)

            letter_color = random.choice(['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'white', 'black', 'magenta', 'gray', (216, 191, 216), (255, 255, 224)])

            text_bbox_font = font.getbbox(letter)
            text_width = text_bbox_font[2] - text_bbox_font[0]
            text_height = text_bbox_font[3] - text_bbox_font[1]

            x0 = (image_size[0] - text_width) // 2
            y0 = (image_size[1] - text_height) // 2

            # Creates text later to draw into Image
            text_layer = Image.new('RGBA', image_size, (0, 0, 0, 0))
            text_draw = ImageDraw.Draw(text_layer)
            text_draw.text((x0, y0), letter, fill=letter_color, font=font, stroke_width=8, stroke_fill=letter_color)

            # Rotate!!!!
            rotation_angle = random.randint(0, 360)
            rotated_text = text_layer.rotate(rotation_angle, expand=True)

            rotated_width, rotated_height = rotated_text.size
            paste_x = (image_size[0] - rotated_width) // 2
            paste_y = (image_size[1] - rotated_height) // 2

            image.paste(rotated_text, (paste_x, paste_y), rotated_text)

            # Get text bbox
            text_mask = text_layer.split()[-1]
            rotated_text_mask = text_mask.rotate(rotation_angle, expand=True)
            text_bbox = rotated_text_mask.getbbox()

            text_bbox_left = paste_x + text_bbox[0]
            text_bbox_top = paste_y + text_bbox[1]
            text_bbox_right = paste_x + text_bbox[2]
            text_bbox_bottom = paste_y + text_bbox[3]

            # Important for making non corrupt image (skips this generation)
            if not is_bbox_valid((text_bbox_left, text_bbox_top, text_bbox_right, text_bbox_bottom), image_size):
                print(f"Skipping image {i + 1} due to out of bounds bounding box")
                continue

            char_center_x = (text_bbox_left + text_bbox_right) / 2 / image_size[0]
            char_center_y = (text_bbox_top + text_bbox_bottom) / 2 / image_size[1]
            char_width = (text_bbox_right - text_bbox_left) / image_size[0]
            char_height = (text_bbox_bottom - text_bbox_top) / image_size[1]

            letter_class_id = letter_classes[letter]
            char_annotation = f"{letter_class_id} {char_center_x:.6f} {char_center_y:.6f} {char_width:.6f} {char_height:.6f}\n"

            images_and_labels.append((image, char_annotation))

            print(f"Character Image {i + 1} generated!")

    return images_and_labels

shape_images_and_labels = generate_shape_images(num_images_per_shape=100)
character_images_and_labels = generate_character_images(num_images_per_char=40)

all_images_and_labels = shape_images_and_labels + character_images_and_labels

random.shuffle(all_images_and_labels)

train_split = 0.8 # Train val split
num_train = int(len(all_images_and_labels) * train_split)
train_images_and_labels = all_images_and_labels[:num_train]
val_images_and_labels = all_images_and_labels[num_train:]

for idx, (image, annotation) in enumerate(train_images_and_labels): # Train split
    image_path = os.path.join(images_dir, f"image_{idx}.png")
    label_path = os.path.join(labels_dir, f"image_{idx}.txt")

    rgb_image = image.convert("RGB")
    rgb_image.save(image_path)

    with open(label_path, "w") as f:
        f.write(annotation)

for idx, (image, annotation) in enumerate(val_images_and_labels): # Val split
    image_path = os.path.join(test_images_dir, f"image_{idx}.png")
    label_path = os.path.join(test_labels_dir, f"image_{idx}.txt")

    rgb_image = image.convert("RGB")
    rgb_image.save(image_path)

    with open(label_path, "w") as f:
        f.write(annotation)

import os
import cv2
import math

# Paths
image_dir = 'images'
label_dir = 'runs/detect/predict4/labels'
output_dir = 'output'
txt_output_dir = 'output/txt'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(txt_output_dir, exist_ok=True)

def calculate_orientation(box_w, box_h):
    """Estimate orientation based on aspect ratio"""
    aspect_ratio = box_w / box_h
    if aspect_ratio > 1.2:
        return "Horizontal"
    elif aspect_ratio < 0.8:
        return "Vertical"
    else:
        return "Square"

def estimate_xyz(x, y, box_w, box_h, width, height, focal_length=700):
    """Estimate 3D x, y, z coordinates (simple pinhole camera model)"""
    depth = (width / box_w) * focal_length
    x_real = (x - 0.5) * width * depth / focal_length
    y_real = (y - 0.5) * height * depth / focal_length
    return round(x_real, 2), round(y_real, 2), round(depth, 2)

def calculate_angle(x, y, box_w, box_h, width, height):
    """Calculate the angle of the box relative to the image plane"""
    dx = (x - 0.5) * width
    dy = (y - 0.5) * height
    angle = math.degrees(math.atan2(dy, dx))
    return round(angle, 2)

def draw_boxes(image_path, label_path, output_path, txt_output_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    with open(label_path, 'r') as f:
        lines = f.readlines()

    box_count = 0
    box_data = []

    for idx, line in enumerate(lines):
        values = line.split()
        if len(values) == 5:
            cls, x, y, box_w, box_h = map(float, values)
            conf = 1.0
        else:
            cls, x, y, box_w, box_h, conf = map(float, values)

        # Convert to pixel coordinates
        x1 = int((x - box_w / 2) * width)
        y1 = int((y - box_h / 2) * height)
        x2 = int((x + box_w / 2) * width)
        y2 = int((y + box_h / 2) * height)

        center_x = int(x * width)
        center_y = int(y * height)

        # Draw bounding box
        color = (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Calculate parameters
        orientation = calculate_orientation(box_w, box_h)
        x_real, y_real, z_real = estimate_xyz(x, y, box_w, box_h, width, height)
        angle = calculate_angle(x, y, box_w, box_h, width, height)

        # Assign box ID
        box_id = f'Box {idx + 1}'

        # Draw only the box number and angle
        label = f'{box_id}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Store box details in text
        box_info = (
            f'{box_id}\n'
            f'Class: {int(cls)}\n'
            f'Confidence: {conf:.2f}\n'
            f'Orientation: {orientation}\n'
            f'Angle: {angle}deg\n'
            f'X: {x_real}, Y: {y_real}, Z: {z_real}\n'
        )
        box_data.append(box_info)

        # Print box details in console
        print(box_info)

        box_count += 1

    # Save image with box IDs and angles
    cv2.imwrite(output_path, image)
    print(f'{os.path.basename(image_path)} - Total Boxes: {box_count}\n')

    # Save detailed box info to a text file
    with open(txt_output_path, 'w') as f:
        f.write('\n'.join(box_data))

# Process all images
for filename in os.listdir(image_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        output_path = os.path.join(output_dir, filename)
        txt_output_path = os.path.join(txt_output_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))

        if os.path.exists(label_path):
            draw_boxes(image_path, label_path, output_path, txt_output_path)
        else:
            print(f'Label not found for {filename}')

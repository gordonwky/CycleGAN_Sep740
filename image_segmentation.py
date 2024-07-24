import torch
from torchvision import models, transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load a pre-trained segmentation model
model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

# Define the transformation
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def segment_image(image_path):
    # Open the image
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    # Move the input to the GPU for faster processing
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0).cpu().numpy()

    return output_predictions

def create_binary_mask(output_predictions, class_index):
    mask = output_predictions == class_index
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    return mask_image

def apply_mask(image_path, mask):
    image = Image.open(image_path).convert("RGB")
    image.show(title="Original Image")
    # mask = Image.fromarray((np.array(mask_image) / 255).astype(np.uint8))
    mask.show(title="masked Image")
    # Apply the mask to the image
    masked_image = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), mask)
    masked_image.show(title="Combined Image")
    return masked_image

# Assuming you have a function to load your CycleGAN model

# model = load_cyclegan_model()
# turn masked image into green 
def test_transform(masked_image):
    return masked_image
def recombine_images(original_image_path, transformed_segment, mask_image):
    original_image = Image.open(original_image_path).convert("RGB")
    mask = Image.fromarray((np.array(mask_image) / 255).astype(np.uint8))

    # Resize the transformed segment to match the original image size
    transformed_segment = transformed_segment.resize(original_image.size, Image.BICUBIC)

    # Recombine the transformed segment with the original image
    recombined_image = Image.composite(transformed_segment, original_image, mask)
    return recombined_image

# Apply the segmentation to an image
image_path = '/Users/kimyingwong/CycleGAN_Sep740/Unknown.jpeg'
output_predictions = segment_image(image_path)

# Visualize the raw output
plt.imshow(output_predictions)
plt.colorbar()
plt.title("Segmentation Output")
plt.show()
print(np.unique(output_predictions))
# Assuming 'horse' class has the index 13
horse_class_index = 13
mask_image = create_binary_mask(output_predictions, horse_class_index)

# mask_image.show()

# # Transform the segmented part
# cycle_transform = transforms.Compose([
#     transforms.Resize((256, 256)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

masked_image = apply_mask(image_path, mask_image)


# masked_image.show()
transformed_segment = test_transform(masked_image)

# transformed_segment = transform_segmented_part(masked_image, cycle_transform, model)
# transformed_segment.show()

# # Recombine the images
recombined_image = recombine_images(image_path, transformed_segment, mask_image)
recombined_image.show()

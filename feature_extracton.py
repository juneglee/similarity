from PIL import Image
import torch
from torchvision import transforms

# Model loading.
model = torch.jit.load('weight/inception_v3.pt')
model.eval()
embedding_fn = model

# Load image and extract image embedding
def embedding(image_path):
    input_image = Image.open(image_path).convert("RGB")
    convert_to_tensor = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.PILToTensor()
    ])
    input_tensor = convert_to_tensor(input_image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        embedding = torch.flatten(embedding_fn(input_batch)[0]).cpu().data.numpy()

    return embedding

if __name__ == '__main__':
    img_path = 'images/dog.jpeg'
    embedding(img_path)
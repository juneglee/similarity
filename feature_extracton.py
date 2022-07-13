from PIL import Image
import torch
from torchvision import transforms
from similarity import cosine_similarity, euclidean

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
    target_path = 'images/cat.jpeg'
    input_path = 'images/dog.jpeg'
    target_feature = embedding(target_path)
    input_feature = embedding(input_path)
    cos_sim = cosine_similarity(target_feature, input_feature)
    euc_di = euclidean(target_feature, input_feature)

    print(f'target : {target_path.split("/")[-1]}, input : {input_path.split("/")[-1]}, Cosine similarity : {cos_sim}')
    print(f'target : {target_path.split("/")[-1]}, input : {input_path.split("/")[-1]}, euclidean distance : {euc_di}')



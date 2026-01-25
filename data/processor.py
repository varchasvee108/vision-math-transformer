import torch
from torchvision import transforms
from PIL import Image

VOCAB = ["<SOS>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "<EOS> ", "<PAD>"]
VOCAB_SIZE = len(VOCAB)

stoi = {ch: i for i, ch in enumerate(VOCAB)}
itos = {i: ch for i, ch in enumerate(VOCAB)}

PAD_ID = stoi["<PAD>"]
SOS_ID = stoi["<SOS>"]
EOS_ID = stoi["<EOS>"]


class VisionMathProcessor:
    def __init__(self, image_size):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def preprocess(self, image):
        return self.image_transform(image)

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        chars = []
        for t in token_ids:
            if t in (EOS_ID, PAD_ID):
                break
            chars.append(itos[t])
        return "".join(chars)

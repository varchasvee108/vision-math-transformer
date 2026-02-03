import torch
from torchvision import transforms
from PIL import Image
from typing import Tuple


class VisionProcessor:
    def __init__(self, image_size: Tuple, vocab):
        self.vocab = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}

        self.pad_id = self.stoi["<PAD>"]
        self.sos_id = self.stoi["<SOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.vocab_size = len(vocab)

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def process_image(self, image):
        return self.image_transform(image)

    def encode(self, tokens):
        tokens = [self.sos_id] + [self.stoi[token] for token in tokens] + [self.eos_id]
        return tokens

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        chars = []

        for t in token_ids:
            if t == self.sos_id:
                continue
            if t in (self.eos_id, self.pad_id):
                break
            chars.append(self.itos[t])
        return "".join(chars)

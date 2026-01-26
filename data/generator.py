import os
import json
import random
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import argparse
from data.processor import VisionProcessor
from core.config import Config


def generate_single_image(
    size: tuple, max_digits: int = 4, processor: VisionProcessor = None
):
    width, height = size
    img = Image.new("L", size=size, color=255)
    draw = ImageDraw.Draw(img)

    a = (random.randint(0, 999),)
    b = random.randint(0, 999)

    result = a + b

    var_text = f"a = {a} \n b = {b}"
    query_text = "Find: a + b"

    try:
        font = ImageFont.truetype("aria.ttf", 32)
    except:
        font = ImageFont.load_default()

    rand_x = random.randit(10, 50)
    rand_y = random.randint(10, 50)

    draw.text((rand_x, rand_y), text=var_text, fill=0, font=font)
    center_y = height // 2 + random.randint(-20, 20)
    draw.text((rand_x, center_y), text=query_text, fill=0, font=font)

    answer_str = str(result)
    answer_tokens = processor.encode(answer_str)

    while len(answer_tokens) < max_digits + 2:
        answer_tokens.append(processor.pad_id)

    return img, answer_tokens, result

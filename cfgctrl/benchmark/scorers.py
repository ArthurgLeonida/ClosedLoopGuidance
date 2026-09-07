"""Lazy, explicit implementations of the paper's per-image metric families.

Scores retain native units: CLIP/PickScore/HPS cosine, ImageReward normalized
reward, aesthetic regressor output, MPS overall logit, and VQAScore probability.
Never apply a softmax across unrelated prompts or a one-image softmax.
"""
from contextlib import nullcontext
from pathlib import Path
import sys

import torch
from PIL import Image


def paired_cosine(images, texts):
    images = torch.nn.functional.normalize(images.float(), dim=-1)
    texts = torch.nn.functional.normalize(texts.float(), dim=-1)
    if images.shape != texts.shape:
        raise ValueError("metric embeddings must have one text per image")
    return (images * texts).sum(dim=-1)


class HFSimilarity:
    def __init__(self, name, device, options):
        from transformers import AutoModel, AutoProcessor
        self.device = device
        checkpoint = options.get("checkpoint", "openai/clip-vit-large-patch14" if name == "clip"
                                 else "yuvalkirstain/PickScore_v1")
        processor = options.get("processor", "openai/clip-vit-large-patch14" if name == "clip"
                                else "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained(checkpoint, revision=options.get("revision", "main")).eval().to(device)
        self.processor = AutoProcessor.from_pretrained(processor, revision=options.get("processor_revision", "main"))

    def __call__(self, records):
        with torch.inference_mode():
            pictures = []
            for r in records:
                with Image.open(r["path"]) as im:
                    pictures.append(im.convert("RGB"))
            images = self.processor(images=pictures, return_tensors="pt").to(self.device)
            text = self.processor(text=[r["record"]["prompt"] for r in records], padding=True,
                                  truncation=True, max_length=77, return_tensors="pt").to(self.device)
            return paired_cosine(self.model.get_image_features(**images),
                                 self.model.get_text_features(**text)).cpu().tolist()


class Aesthetic:
    def __init__(self, device, options):
        import clip
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)
        self.device = device
        checkpoint = Path(options["checkpoint"])
        # Same layer order and checkpoint keys as the official predictor.
        self.head = torch.nn.Sequential(torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2),
                                        torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
                                        torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
                                        torch.nn.Linear(64, 16), torch.nn.Linear(16, 1))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.head.load_state_dict({k.removeprefix("layers."): v for k, v in state.items()})
        self.head.eval().to(device)

    def __call__(self, records):
        pictures = []
        for r in records:
            with Image.open(r["path"]) as im:
                pictures.append(self.preprocess(im.convert("RGB")))
        with torch.inference_mode():
            features = self.model.encode_image(torch.stack(pictures).to(self.device)).float()
            features = torch.nn.functional.normalize(features, dim=-1)
            return self.head(features).flatten().cpu().tolist()


class ImageReward:
    def __init__(self, device, options):
        import ImageReward as reward
        self.model = reward.load(options.get("checkpoint", "ImageReward-v1.0"), device=device)

    def __call__(self, records):
        with torch.inference_mode():
            return [float(self.model.score(r["record"]["prompt"], r["path"])) for r in records]


class HPS:
    def __init__(self, name, device, options):
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        from huggingface_hub import hf_hub_download
        self.device = device
        # Load once per metric worker, not once per image; v2 and v2.1 are separate workers.
        self.model, _, self.preprocess = create_model_and_transforms(
            "ViT-H-14", pretrained=None, precision="fp32", device=device, output_dict=True,
            with_score_predictor=False, with_region_predictor=False)
        self.tokenizer = get_tokenizer("ViT-H-14")
        filename = "HPS_v2_compressed.pt" if name == "hpsv2" else "HPS_v2.1_compressed.pt"
        checkpoint = options.get("checkpoint") or hf_hub_download(
            "xswu/HPSv2", filename, revision=options.get("revision", "main"))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()

    def __call__(self, records):
        pictures = []
        for r in records:
            with Image.open(r["path"]) as im:
                pictures.append(self.preprocess(im.convert("RGB")))
        text = self.tokenizer([r["record"]["prompt"] for r in records]).to(self.device)
        autocast = torch.autocast("cuda") if self.device.startswith("cuda") else nullcontext()
        with torch.inference_mode(), autocast:
            features = self.model(torch.stack(pictures).to(self.device), text)
            return paired_cosine(features["image_features"], features["text_features"]).cpu().tolist()


MPS_CONDITION = ("light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, "
                 "limbs, structure, instance, texture, quantity, attributes, position, number, "
                 "location, word, things.")


class MPS:
    def __init__(self, device, options):
        from transformers import AutoTokenizer, CLIPImageProcessor
        # Official MPS distribution serializes its model class, requiring its source tree.
        repository = Path(options["repo"]).resolve()
        if not repository.is_dir():
            raise ValueError(f"MPS source repository is missing: {repository}")
        sys.path.insert(0, str(repository))
        self.model = torch.load(options["checkpoint"], map_location="cpu", weights_only=False).eval().to(device)
        checkpoint = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.processor = CLIPImageProcessor.from_pretrained(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.device = device

    def tokenize(self, text):
        return self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length",
                              truncation=True, return_tensors="pt").input_ids.to(self.device)

    def __call__(self, records):
        values = []
        with torch.inference_mode():
            for r in records:
                with Image.open(r["path"]) as im:
                    image = self.processor(im.convert("RGB"), return_tensors="pt")["pixel_values"].to(self.device)
                # The official paired interface embeds two candidates. Duplicating one
                # obtains its raw score; softmax here would incorrectly always give 0.5.
                text, first, _ = self.model(self.tokenize(r["record"]["prompt"]),
                                           torch.cat([image, image]), self.tokenize(MPS_CONDITION))
                score = self.model.logit_scale.exp() * paired_cosine(first, text)
                values.append(float(score.item()))
        return values


class VQAScore:
    def __init__(self, device, options):
        from importlib.metadata import version
        if version("t2v-metrics") != "1.1":
            raise RuntimeError("use the pinned t2v-metrics==1.1 image-evaluation environment for CLIP-FlanT5 VQAScore")
        import t2v_metrics
        self.model = t2v_metrics.VQAScore(model=options.get("checkpoint", "clip-flant5-xxl"), device=device)

    def __call__(self, records):
        data = [dict(images=[r["path"]], texts=[r["record"]["prompt"]]) for r in records]
        with torch.inference_mode():
            result = self.model.batch_forward(dataset=data, batch_size=len(data))
        if tuple(result.shape) != (len(records), 1, 1):
            raise RuntimeError(f"unexpected VQAScore shape: {tuple(result.shape)}")
        return result[:, 0, 0].cpu().tolist()


def load_scorer(name, device, options):
    if name in ("clip", "pickscore"):
        return HFSimilarity(name, device, options)
    if name in ("hpsv2", "hpsv21"):
        return HPS(name, device, options)
    factory = {"aesthetic": Aesthetic, "imagereward": ImageReward, "mps": MPS, "vqascore": VQAScore}
    return factory[name](device, options)

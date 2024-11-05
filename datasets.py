import av
import os
import json
from PIL import Image
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from io import BytesIO
import base64

TO_LOAD_IMAGE: Dict[str, bool] = {
    "llava-1.5": True,
    "llava-1.6": True,
    "llava-interleave": True,
    "llava-next-video": True,
    "qwen-vl": False,
    "phi3-v": True,
    "qwen2-vl": True,
}


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

# /home/pcarragh/dev/webqa/UniVL-DR/data
data_root = "/data/nikitha/VQA_data"
with open(f"{data_root}/imgs.lineidx", "r") as fp_lineidx:
    lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]

# def read_webqa_image(image_id, lineidx=lineidx):
#     try:
#         with open("/home/pcarragh/dev/webqa/UniVL-DR/data/imgs.tsv", "r") as fp:
#             fp.seek(lineidx[int(image_id)%10000000])
#             imgid, img_base64 = fp.readline().strip().split('\t')
#         assert int(image_id) == int(imgid), f'{image_id} {imgid}'
#         im = Image.open(BytesIO(base64.b64decode(img_base64)))
#         return im
#     except Exception as e:
#         # generation
#         return Image.open(image_file).convert("RGB")
        

# # def load_image(image_file):
# #     if image_file.startswith("http") or image_file.startswith("https"):
# #         response = requests.get(image_file)
# #         image = Image.open(BytesIO(response.content)).convert("RGB")
# #     else:
# #         image = Image.open(image_file).convert("RGB")
# #     return image


# def load_images(image_files, webqa=False):
#     out = []
#     for image_file in image_files:
#         if webqa:
#             image = read_webqa_image(image_file)
#         else:
#             image = Image.open(image_file).convert("RGB")
#         out.append(image)
#     return out

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning 
    which is generalized enough to handle both images and videos.
    """

    def __init__(
        self, 
        data_path: str, 
        model_family_id: str,
        image_folder: Optional[str] = None,
        video_folder: Optional[str] = None,
        num_frames: int = 8,
        user_key: str = "human",
        assistant_key: str = "gpt",
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        self.image_folder = image_folder
        self.video_folder = video_folder
        self.num_frames = num_frames
        self.load_image = TO_LOAD_IMAGE[model_family_id]
        self.user_key = user_key
        self.assistant_key = assistant_key

        self.is_text_only = [
            "image" not in source and "video" not in source
            for source in self.list_data_dict
        ]

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, List]:      
        source = self.list_data_dict[i]

        images = []
        if "image" in source:
            # here we do not do any image preprocessing but rather
            # let the processor handle everything
            # in some cases this may cause slight differences
            # but should totally be fine (e.g., official llava-1.5 does padding,
            # but llava-1.5-hf (huggingface's implementation) does not)
            if isinstance(source["image"], list):
                image_sources = source["image"]
            elif isinstance(source["image"], str):
                image_sources = [source["image"]]
            else:
                raise ValueError(f"Invalid image source type: {type(source['image'])}")
            
            for image_id in image_sources:              
                try:
                    image_path_int = int(image_id)
                except:
                    image_path_int = None
                    
                if image_path_int:
                    with open(f"{data_root}/imgs.tsv", "r") as fp:
                        fp.seek(lineidx[int(image_id)%10000000])
                        imgid, img_base64 = fp.readline().strip().split('\t')
                    assert int(image_id) == int(imgid), f'{image_id} {imgid}'
                    images.append(Image.open(BytesIO(base64.b64decode(img_base64))))
                else:
                    if self.image_folder is not None:
                        image_id = os.path.join(self.image_folder, image_id)
                    images.append(Image.open(image_id).convert("RGB"))
                # images.append(
                #     Image.open(image_path).convert("RGB")
                #     if self.load_image else image_path
                # )

        videos = []
        if "video" in source:
            if isinstance(source["video"], list):
                video_sources = source["video"]
            elif isinstance(source["video"], str):
                video_sources = [source["video"]]
            else:
                raise ValueError(f"Invalid video source type: {type(source['video'])}")

            num_frames = [self.num_frames] * len(video_sources)

            for video_path, cur_num_frames in zip(video_sources, num_frames):
                if self.video_folder is not None:
                    video_path = os.path.join(self.video_folder, video_path)
                
                container = av.open(video_path)
                total_frames = container.streams.video[0].frames
                indices = np.arange(0, total_frames, total_frames / cur_num_frames).astype(int)
                clip = read_video_pyav(container, indices)

                videos.append(clip)
        
        system_prompt = None
        if "system_prompt" in source:
            system_prompt = source["system_prompt"]

        convs = []
        assert len(source["conversations"]) > 0, "No conversations found"
        for i, conv in enumerate(source["conversations"]):
            assert conv["from"] == (self.user_key if i % 2 == 0 else self.assistant_key), "Invalid conversation"
            convs.append(conv["value"])
        assert len(convs) % 2 == 0, "Odd number of conversations"
        
        return dict(
            images=images,
            videos=videos,
            conversations=convs,
            system_prompt=system_prompt
        )
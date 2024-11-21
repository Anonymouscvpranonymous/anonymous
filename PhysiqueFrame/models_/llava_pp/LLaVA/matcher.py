import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, \
    IMAGE_PLACEHOLDER
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
import requests
from PIL import Image
from io import BytesIO
import re
import torch.nn as nn


# import option
#
# opt = option.init()
# device = torch.device("cuda:{}".format(opt.llava_gpu_id))


class LLaVA_Matcher(nn.Module):
    def __init__(self, model_path):
        super(LLaVA_Matcher, self).__init__()
        disable_torch_init()
        self.model_path = model_path
        self.model_name = get_model_name_from_path(model_path)
        self.image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        self.conv_mode = "phi3_instruct"
        self.temperature = 0.2
        self.top_p = 0.7
        self.num_beams = 1
        self.max_new_tokens = 512
        self.qs_template = "A user preference are that \"{}\". Please analyze the image and " \
                           "summarize how this preference affects the user's evaluation of the " \
                           "image in the following 5 dimensions: 1. physique expressiveness, 2. " \
                           "physique style (such as extroverted and introverted), 3. physique shape " \
                           "(such as lean and pretty stacked), 4. physique posture, 5. personality traits. " \
                           "Please provide the degree of influence of this preference effects of the user's " \
                           "evaluation of the image in 5 dimensions (values ranging from 0 to 1, 0.5 " \
                           "represents no impressed or suppressed, 1 represents completely impressed , 0 " \
                           "represents completely suppressed), without any other explaination and content."

    def init_model(self):
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=self.model_path,
            model_base=None,
            model_name=self.model_name,
            # device=device
        )

    def image_parser(self, args):
        out = args.image_file.split(args.sep)
        return out

    def load_image(self, image_file):
        if image_file.startswith("http") or image_file.startswith("https"):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
        return image

    def load_images(self, image_files):
        out = []
        for image_file in image_files:
            image = self.load_image(image_file)
            out.append(image)
        return out

    def prepare_prompt(self, preference_str):
        qs = self.qs_template.format(preference_str)
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, self.image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = self.image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt

    def get_score_list(self, text):

        pattern = r"\d+\.\s\w+.*?:\s([0-1]\.\d)"
        scores = re.findall(pattern, text)
        # 将分数转换为浮点数并存储在列表中
        scores_list = [float(score) for score in scores]
        return scores_list

    def match_and_summarize(self, preference_str, image_path):
        # Summary Favorite
        prompt = self.prepare_prompt(preference_str)
        image_files = [image_path]
        images = self.load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
        )
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        outputs = outputs.replace("<|end|>", "").strip()
        return self.get_score_list(outputs)



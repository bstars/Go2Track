import os
import numpy as np
from typing import List, Callable, Tuple
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
import cv2
import re
import matplotlib.pyplot as plt


from prompt import react_navigation_system_prompt_template 


def np_rgb_img_to_data_url(img, max_size=(384, 384), quality=60):
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img, mode="RGB")
    pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)

    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality, optimize=True)

    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

class ReActAgent:
    def __init__(self, model:str, reasoning="medium") -> None:
        self.model = model
        self.reasoning = reasoning
        self.client = OpenAI(api_key="your-api-key")
        self.response_id = None

    def run(self, user_input:str, fpv_img, third_person_img, execute_command, max_steps=100):
        print("----------------------------------------------------------------------------------")
        print("requesting model ...")
        response = self.client.responses.create(
            model=self.model,
            instructions=react_navigation_system_prompt_template,
            reasoning={ "effort": self.reasoning },
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text", 
                            "text": f"<question>{user_input} The initial third-person-view are attached.</question>"
                        },
                        {
                            "type": "input_image",
                            "image_url": np_rgb_img_to_data_url(third_person_img)
                        } ,
                    ],
                }
            ],
        )
        self.response_id = response.id
        content = response.output_text

        for _ in range(max_steps):
            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1).strip()
                print(f"Thought: {thought}")

            if "<final_answer>" in content:
                return content

            action_match = re.search(
                r"<action>\s*move\(([^,]+),([^,]+),([^)]+)\)\s*</action>",
                content,
                re.DOTALL,
            )

            if not action_match:
                raise RuntimeError(f"No valid action found in model response: {content}")

            vx, vy, omega = map(float, action_match.groups())
            command = np.array([vx, vy, omega], dtype=np.float32)
            print(f"action", command)

            fpv_img, third_person_img = execute_command(command)
            print("----------------------------------------------------------------------------------")
            print("requesting model ...")

            response = self.client.responses.create(
                model=self.model,
                reasoning={ "effort": self.reasoning },
                previous_response_id=self.response_id,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text", 
                                "text": "<observation>The new third-person-view after executing the movement are attached.</observation>"
                            },
                            {
                                "type": "input_image",
                                "image_url": np_rgb_img_to_data_url(third_person_img)
                            },
                        ],
                    }
                ],
            )

            content = response.output_text

            


        
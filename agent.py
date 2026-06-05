
import os
import numpy as np
from typing import List, Callable, Tuple
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
import cv2
import re


from prompt import react_navigation_system_prompt_template 


def np_rgb_img_to_data_url(img):
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img, mode="RGB")

    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")

    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

class ReActAgent:
    def __init__(self, model:str) -> None:
        self.model = model
        self.client = OpenAI(api_key="your-api-key")

    def call_model(self, messages):

        print("requesting model ...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        return content

    def run(self, user_input:str, fpv_img, third_person_img, execute_command, max_steps=100):
        messages = [
            {"role": "system", "content": react_navigation_system_prompt_template},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<question>{user_input} The initial third-person-view are attached.</question>"
                    },
                    # {
                    #     "type": "image_url",
                    #     "image_url": {"url": np_rgb_img_to_data_url(fpv_img)}
                    # },
                    {
                        "type": "image_url",
                        "image_url": {"url": np_rgb_img_to_data_url(third_person_img)}
                    }
                ]
            }
        ]

        for _ in range(max_steps):
            print("----------------------------------------------------------------------------------------")
            content = self.call_model(messages)

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

            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "<observation>"
                            "The new third-person-view after executing the movement are attached."
                            "</observation>"
                        )
                    },
                    # {
                    #     "type": "image_url",
                    #     "image_url": {"url": np_rgb_img_to_data_url(fpv_img)}
                    # },
                    {
                        "type": "image_url",
                        "image_url": {"url": np_rgb_img_to_data_url(third_person_img)}
                    }
                ]
            })

        return "<final_answer>Stopped because max_steps was reached.</final_answer>"

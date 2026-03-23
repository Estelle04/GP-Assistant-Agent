import os
import json
import base64
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Image_processing:
    def __init__(self):
        self.image_folder = os.path.join(BASE_DIR, 'images')
        self.output_path = os.path.join(BASE_DIR, 'image_result.json')
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def encode_image(self, image):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def image_query(self, individual_image, filename):

        system_prompt = (
            "You are a GP assistant analysing a patient-submitted image as part of a pre-consultation triage.\n"
            "The image may be any clinical photo taken by the patient, or an assessment that the patient has been through.\n"
            "Your role is to provide a preliminary visual assessment only — not a definitive diagnosis.\n\n"

            "Respond with ONLY a single JSON object. No text before or after. No markdown. No code fences.\n"
            "Use EXACTLY this structure:\n"
            "{\n"
            '  "image_type": "ECG / skin lesion / rash / swelling / other",\n'
            '  "visual_findings": ["finding 1", "finding 2"],\n'
            '  "preliminary_assessment": "Brief summary of what the image suggests clinically",\n'
            '  "red_flags": ["urgent concern 1"] or [],\n'
            '  "recommended_follow_up": ["suggested next step 1", "suggested next step 2"]\n'
            "}\n\n"

            "Rules:\n"
            "- visual_findings: objective observations only, e.g. 'ST elevation in leads V1-V4', 'asymmetric pigmented lesion'\n"
            "- preliminary_assessment: 1-2 sentences, hedged language e.g. 'findings are consistent with...', 'may suggest...'\n"
            "- red_flags: only include if something is genuinely urgent, otherwise return empty list []\n"
            "- recommended_follow_up: practical next steps e.g. 'urgent cardiology referral', 'dermoscopy', 'repeat ECG'\n"
            "- Never give a definitive diagnosis — this is a triage aid for the GP only\n"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{individual_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            }
        ]

        response = self.client.chat.completions.create(
            model="Qwen/Qwen3.5-27B-FP8",
            messages=messages,
            temperature=1.0,
            top_p=0.95,
            presence_penalty=1.5,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }, 
        )

        raw = response.choices[0].message.content
        print(f"{filename}, Done!")
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(clean)
    
    def main(self):
        all_results = []

        image_files = [
            f for f in os.listdir(self.image_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        for filename in image_files:
            full_path = os.path.join(self.image_folder, filename)
            base64_image = self.encode_image(full_path)
            result = self.image_query(base64_image, filename)
            result['patient name'] = os.path.splitext(filename)[0]
            all_results.append(result)

        with open(self.output_path, 'w') as result_json:
            json.dump(all_results, result_json, indent=4)

        print('All Done')


processor = Image_processing()
processor.main()
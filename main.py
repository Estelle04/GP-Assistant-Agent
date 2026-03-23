import os
import json
import pandas as pd
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv


load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class LLM_process:
    def __init__(self):
        csv_path = os.path.join(BASE_DIR, 'Synthetic_patient_data.csv')
        self.output_path = os.path.join(BASE_DIR, 'text_result.json')
        self.data = pd.read_csv(csv_path)
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_API_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def date_time_convert(self, date_str, time_str):
        start_time_str = time_str.split("-")[0].strip()
        period = time_str.split(" ")[-1].strip()
        combined = f"{date_str} {start_time_str} {period}"
        return datetime.strptime(combined, "%d/%m/%Y %I:%M %p")

    def llm_input(self, individual_data):
        system_prompt = (
            "You are a GP assistant. Please diagnose the given patient base on the given information and give your opinion.\n"
            "Consider how age, gender, and race influence the likelihood of each diagnosis.\n"
            "Consider allergies for each diagnosis, avoid those when prescribe.\n"
            "Respond with ONLY a single JSON object. No text before or after. No markdown. No code fences.\n"

            "Use EXACTLY this structure:\n"
            "{\n"
            '  "symptoms_summary": ["symptom 1", "symptom 2"],\n'
            '  "recommended_inspection": ["test 1", "test 2"],\n'
            '  "diagnoses": [\n'
            '    {"condition": "Disease name", "probability": "40%"},\n'
            '    {"condition": "Disease name", "probability": "35%"},\n'
            '    {"condition": "Disease name", "probability": "25%"}\n'
            '  ],\n'
            '  "severity": "medium",\n'
            '  "advice": "Your advice text here or See GP"\n'
            "}\n\n"

            "1) Summary of the symptoms, separated by commas\n"
            "2) Recommended Inspection (e.g. MRI, CT, blood test), up to three — if none, state 'Not required'\n"
            "3) Probability of diagnoses, up to three — list them with estimated likelihood.\n"
            "4) Severity: one of 'low', 'medium', 'high', 'critical'\n"
            "5) Advice: If the Severity is low, then generate advices to the patient, including prescriptions. If the Severity is critical, suggest seek emergency care immediately. Otherwise state 'See GP'\n"
            
            "Severity definitions:\n"
            "  - low: self-resolves within 3-5 days, no GP visit needed\n"
            "  - medium: recommended to visit GP\n"
            "  - high: must visit GP and undergo inspection\n"
            "  - critical: life-threatening, seek emergency care immediately\n\n"
        )

        user_query = (
            f"Age: {individual_data['Age']}\n"
            f"Gender: {individual_data['Gender']}\n"
            f"Race: {individual_data['Race']}\n"
            f"Allergy: {individual_data['Allergy']}\n"
            f"Symptoms: {individual_data['Symptom Description']}\n"
        )

        response = self.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                ],
                reasoning_effort="minimal",
                response_format={"type": "json_object"}
        )

        raw = response.choices[0].message.content
        print(individual_data['Patient Name'], 'Done!')
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(clean)
    

    def main(self):
        all_results = []

        for _, row in self.data.iterrows():
            result = self.llm_input(row)

            result['patient_name'] = row['Patient Name']
            result['patient_age'] = row['Age']
            result['patient_gender'] = row['Gender']
            result['patient_race'] = row['Race']
            result['preferred_slot_1'] = row['Preferred Date 1'] + ' ' + row['Preferred Time 1']
            result['preferred_slot_2'] = row['Preferred Date 2'] + ' ' + row['Preferred Time 2']

            all_results.append(result)

        
        return all_results

                        


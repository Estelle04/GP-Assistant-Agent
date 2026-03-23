import os
import json
import heapq
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
severity_order = {'critical': 0, 'high': 1, 'medium': 2}

def parse_slot(slot_str):
    parts  = slot_str.strip().split(' ')
    date   = parts[0]
    time   = parts[1].split('-')[0]
    period = parts[2]
    return datetime.strptime(f"{date} {time} {period}", "%d/%m/%Y %I:%M %p")

class Scheduler:
    def __init__(self, output_path=None):
        self.output_path = output_path or os.path.join(BASE_DIR, 'text_result.json')

    def run(self, patients):
        heap = []
        patient_map = {}
        low_patients = []

        for patient in patients:
            if patient.get('severity') == 'low':
                low_patients.append(patient)
                continue

            priority  = severity_order.get(patient.get('severity'), 99)
            slot_1_dt = parse_slot(patient['preferred_slot_1'])
            heapq.heappush(heap, (priority, slot_1_dt, patient['patient_name']))
            patient_map[patient['patient_name']] = patient

        booked_slots = set()
        scheduled    = []
        unscheduled  = []

        while heap:
            _, _, patient_name = heapq.heappop(heap)
            patient = patient_map[patient_name]

            slot_1_dt = parse_slot(patient['preferred_slot_1'])
            slot_2_dt = parse_slot(patient['preferred_slot_2'])

            if slot_1_dt not in booked_slots:
                booked_slots.add(slot_1_dt)
                patient['assigned_slot'] = patient['preferred_slot_1']
                patient['slot_status']   = 'slot 1 confirmed'
                scheduled.append(patient)

            elif slot_2_dt not in booked_slots:
                booked_slots.add(slot_2_dt)
                patient['assigned_slot'] = patient['preferred_slot_2']
                patient['slot_status']   = 'slot 2 confirmed'
                scheduled.append(patient)

            else:
                patient['assigned_slot'] = None
                patient['slot_status']   = 'unscheduled — both slots taken'
                unscheduled.append(patient)

        for patient in low_patients:
            patient['assigned_slot'] = 'NA'
            patient['slot_status']   = 'NA'

        all_results = scheduled + unscheduled + low_patients
        for patient in all_results:
            patient.pop('preferred_slot_1', None)
            patient.pop('preferred_slot_2', None)

        scheduled.sort(key=lambda p: parse_slot(p['assigned_slot']))
        all_results = scheduled + unscheduled + low_patients

        with open(self.output_path, 'w') as f:
            json.dump(all_results, f, indent=4)

        print(f"\nDone")
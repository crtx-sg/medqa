import time
import random
import datetime
import math
from .ews_analyzer import EWSTrendAnalyzer

class ClinicalDataTools:
    """A collection of low-level tools to simulate clinical data retrieval."""
    def __init__(self, logger):
        self.logger = logger
        self._generate_patient_data()

    def _generate_patient_data(self):
        """Creates a consistent set of synthetic patient data for the session."""
        self.patients = {}
        wards = {"Cardiology": "CARD", "Neurology": "NEURO", "Oncology": "ONC"}
        doctors = ["Dr. Smith", "Dr. Jones", "Dr. Patel", "Dr. Garcia", "Dr. Chen"]
        
        patient_names = [
            "John Doe", "Jane Smith", "Robert Johnson", "Emily Williams", "Michael Brown",
            "Jessica Davis", "David Miller", "Sarah Wilson", "James Moore", "Linda Taylor"
        ]
        
        allergies = [["Penicillin"], ["Sulfa"], ["None"], ["Ibuprofen"], ["Codeine"]]
        diets = ["Low Sodium", "Diabetic", "Regular", "Heart Healthy", "Renal"]
        prescriptions = [
            ["Lisinopril 10mg", "Metformin 500mg"],
            ["Aspirin 81mg", "Atorvastatin 20mg"],
            ["Levothyroxine 50mcg", "Amlodipine 5mg"],
            ["Warfarin 5mg", "Furosemide 40mg"],
            ["Chemotherapy Cycle A", "Ondansetron 8mg"]
        ]

        patient_counter = 0
        for ward, prefix in wards.items():
            for j in range(3): # Create 3 patients per ward
                if patient_counter < len(patient_names):
                    patient_id = f"{prefix}{101+j}"
                    self.patients[patient_id] = {
                        "name": patient_names[patient_counter],
                        "id": patient_id,
                        "ward": ward,
                        "room": f"{prefix[0]}{j+1}",
                        "bed": random.choice(["A", "B"]),
                        "doctor": random.choice(doctors),
                        "protocol": {
                            "prescriptions": random.choice(prescriptions),
                            "diet": random.choice(diets),
                            "allergies": random.choice(allergies)
                        }
                    }
                    patient_counter += 1

    def list_wards(self, params=None):
        self.logger.log("TOOL: Listing all hospital wards.")
        return {"wards": list(set(p['ward'] for p in self.patients.values()))}

    def list_patients_by_ward(self, params):
        ward = params.get("ward")
        self.logger.log(f"TOOL: Listing patients for ward: {ward}")
        patients_in_ward = [p for p in self.patients.values() if p['ward'].lower() == ward.lower()]
        return {"patients": patients_in_ward}

    def get_patient_by_id(self, patient_id):
        return self.patients.get(patient_id)

    def get_patient_protocol(self, params):
        patient_id = params.get("patient_id")
        self.logger.log(f"TOOL: Getting protocol for patient: {patient_id}")
        patient = self.get_patient_by_id(patient_id)
        return patient.get("protocol", {}) if patient else {}

    def get_patient_vitals(self, params):
        patient_id = params.get("patient_id")
        duration_hours = params.get("duration_hours", 24)
        self.logger.log(f"TOOL: Getting vitals for patient {patient_id} over {duration_hours}h.")
        
        now = int(time.time())
        vitals = []
        for i in range(duration_hours * 4): # Every 15 mins
            ts = now - i * 900
            vitals.append({
                "timestamp": ts,
                "hr": random.randint(65, 85),
                "rr": random.randint(14, 20),
                "bp": f"{random.randint(110, 130)}/{random.randint(70, 85)}",
                "temp": round(random.uniform(97.5, 99.0), 1),
                "spo2": random.randint(96, 99),
                "loc": "Alert",
                "fluid_output": random.randint(30, 60)
            })
        return {"patient_id": patient_id, "vitals": vitals}

    def get_patient_last_ecg(self, params):
        patient_id = params.get("patient_id")
        self.logger.log(f"TOOL: Getting last ECG for patient: {patient_id}")
        fs = 250
        t = [i / fs for i in range(12 * fs)]
        signal = [math.sin(2 * math.pi * 1.0 * tt) * (1 + 0.05 * random.uniform(-1, 1)) for tt in t]
        return {
            "patient_id": patient_id,
            "timestamp": int(time.time()) - random.randint(300, 600),
            "lead": "II",
            "fs": fs,
            "t": t,
            "signal": signal
        }

    def get_patient_image_study(self, params):
        patient_id = params.get("patient_id")
        self.logger.log(f"TOOL: Getting image study for patient: {patient_id}")
        return {"patient_id": patient_id, "dicom_uid": f"1.2.840.{random.randint(10000, 99999)}"}

    def get_patient_active_alarms(self, params):
        patient_id = params.get("patient_id")
        self.logger.log(f"TOOL: Getting active alarms for patient: {patient_id}")
        return {
            "patient_id": patient_id,
            "alarms": [
                {"timestamp": int(time.time()) - 150, "alarm": "High Heart Rate", "value": "125 bpm"},
                {"timestamp": int(time.time()) - 900, "alarm": "Low SpO2", "value": "91%"}
            ]
        }

    def get_all_patients_last_ews(self, params=None):
        self.logger.log("TOOL: Getting last EWS for all patients.")
        ews_data = []
        for patient_id in self.patients:
            score = random.randint(0, 5)
            ews_data.append({
                "patient_id": patient_id,
                "patient_name": self.patients[patient_id]['name'],
                "ews_score": score,
                "timestamp": int(time.time()) - random.randint(60, 1800)
            })
        return {"patients_ews": ews_data}

    def get_critical_patients(self, params=None):
        self.logger.log("TOOL: Identifying critical patients via EWS trend.")
        critical_list = []
        for patient_id, patient_info in self.patients.items():
            if random.random() < 0.2: # 20% chance of being critical
                critical_list.append({
                    "patient_id": patient_id,
                    "name": patient_info['name'],
                    "room": patient_info['room'],
                    "bed": patient_info['bed'],
                    "status": "Deteriorating",
                    "confidence": "High"
                })
        return {"critical_patients": critical_list}

    def get_patient_order_for_rounds(self, params):
        ward = params.get("ward")
        self.logger.log(f"TOOL: Getting patient round order for ward: {ward}")
        patients_in_ward = [p for p in self.patients.values() if p['ward'].lower() == ward.lower()]
        random.shuffle(patients_in_ward)
        return {"rounding_order": [p['name'] for p in patients_in_ward]}


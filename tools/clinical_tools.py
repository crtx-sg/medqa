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
        # ... (This method remains the same)
        self.patients = {}
        wards = {"Cardiology": "CARD", "Neurology": "NEURO", "Oncology": "ONC"}
        doctors = ["Dr. Smith", "Dr. Jones", "Dr. Patel", "Dr. Garcia", "Dr. Chen"]
        patient_names = [ "John Doe", "Jane Smith", "Robert Johnson", "Emily Williams", "Michael Brown", "Jessica Davis", "David Miller", "Sarah Wilson", "James Moore", "Linda Taylor" ]
        patient_counter = 0
        for ward, prefix in wards.items():
            for j in range(3):
                if patient_counter < len(patient_names):
                    patient_id = f"{prefix}{101+j}"
                    self.patients[patient_id] = { "name": patient_names[patient_counter], "id": patient_id, "ward": ward, "room": f"{prefix[0]}{j+1}", "bed": random.choice(["A", "B"]), "doctor": random.choice(doctors), "medications": {"Lisinopril": "10mg daily", "Metformin": "500mg twice daily"}, "history": {"Hypertension", "Type 2 Diabetes"}, }
                    patient_counter += 1

    def _find_patient(self, params):
        # ... (This helper method remains the same)
        patient_id = params.get("patient_id")
        patient_name = params.get("patient_name")
        ward_name = params.get("ward_name")
        if patient_id and patient_id in self.patients: return self.patients[patient_id]
        if patient_name:
            for p in self.patients.values():
                if p['name'].lower() == patient_name.lower(): return p
        if ward_name:
            for p in self.patients.values():
                if p['ward'].lower() == ward_name.lower(): return p
        return None

    def get_patient_info(self, params: dict) -> dict:
        # ... (This function remains the same)
        self.logger.log(f"TOOL: Getting single patient info with params: {params}")
        patient = self._find_patient(params)
        if not patient: return {"error": "Patient not found."}
        fs = 250
        t = [i / fs for i in range(12 * fs)]
        ecg_signal = [math.sin(2 * math.pi * 1.0 * tt) * (1 + 0.05 * random.uniform(-1, 1)) for tt in t]
        return { "patient_name": patient["name"], "patient_id": patient["id"], "ward_name": patient["ward"], "room": patient["room"], "bed": patient["bed"], "doctor_name": patient["doctor"], "medications": patient["medications"], "history": list(patient["history"]), "vitals": { "hr": random.randint(65, 85), "rr": random.randint(14, 20), "bp": f"{random.randint(110, 130)}/{random.randint(70, 85)}", "temperature": round(random.uniform(97.5, 99.0), 1), "spo2": random.randint(96, 99), "ews_score": random.randint(1, 4) }, "ecg": ecg_signal, "alarms": {} }

    def get_all_patients_info(self, params: dict) -> dict:
        # ... (This function remains the same)
        self.logger.log("TOOL: Getting all patients info snapshot.")
        response = { "patients": [], "patient_id": [], "ward_name": [], "room": [], "bed": [], "doctor_name": [], "vitals": [] }
        now = int(time.time())
        for pid, pdata in self.patients.items():
            response["patients"].append(pdata["name"])
            response["patient_id"].append(pid)
            response["ward_name"].append(pdata["ward"])
            response["room"].append(pdata["room"])
            response["bed"].append(pdata["bed"])
            response["doctor_name"].append(pdata["doctor"])
            response["vitals"].append({ "timestamp": now - random.randint(60, 300), "hr": random.randint(65, 85), "rr": random.randint(14, 20), "bp": f"{random.randint(110, 130)}/{random.randint(70, 85)}", "temperature": round(random.uniform(97.5, 99.0), 1), "spo2": random.randint(96, 99), "ews_score": random.randint(1, 4) })
        return response

    def get_patient_alarms(self, params: dict) -> dict:
        # ... (This function remains the same)
        self.logger.log(f"TOOL: Getting patient alarms with params: {params}")
        patient = self._find_patient(params)
        if not patient: return {"error": "Patient not found."}
        alarms = []
        for i in range(random.randint(1, 3)):
            alarms.append({ "timestamp_value": int(time.time()) - random.randint(60, 3600), "alarm_text": random.choice(["Critical HR Alert", "Low SPO2 Alert", "High BP Alert"]), "vitals": { "hr": random.randint(100, 140), "rr": random.randint(22, 28), "bp": f"{random.randint(140, 160)}/{random.randint(90, 100)}", "temperature": round(random.uniform(99.0, 101.0), 1), "spo2": random.randint(88, 94) }, "ecg": [random.uniform(-0.5, 0.5) for _ in range(100)] })
        return { "patient_name": patient["name"], "patient_id": patient["id"], "ward_name": patient["ward"], "room": patient["room"], "bed": patient["bed"], "doctor_name": patient["doctor"], "alarms": alarms }
    
    def get_patient_image_study(self, params: dict) -> dict:
        # ... (This function remains the same)
        self.logger.log(f"TOOL: Getting image study with params: {params}")
        patient = self._find_patient(params)
        if not patient: return {"error": "Patient not found."}
        return { "patient_name": patient["name"], "patient_id": patient["id"], "study_type": "Chest X-Ray", "dicom_uid": f"1.2.840.{random.randint(10000, 99999)}" }

    def get_critical_patients(self, params=None) -> dict:
        # ... (This function remains the same)
        self.logger.log("TOOL: Identifying critical patients based on MEWS policy.")
        critical_list = []
        analyzer = EWSTrendAnalyzer()
        now = datetime.datetime.now()
        for patient_id, patient_info in self.patients.items():
            timestamps = [now - datetime.timedelta(hours=i) for i in range(6)]
            if random.random() < 0.3: ews_values = [3, 4, 4, 5, 6, 7]
            else: ews_values = [2, 3, 2, 3, 2, 3]
            analysis = analyzer.analyze_trend(timestamps, ews_values)
            current_score = ews_values[-1]
            is_critical = False
            reason = ""
            if analysis.get('patient_status') == 'deteriorating':
                is_critical = True
                reason = f"Deteriorating Trend (Confidence: {analysis.get('confidence')})"
            if current_score >= 7:
                is_critical = True
                reason = f"High MEWS Score ({current_score})"
            if is_critical:
                critical_list.append({ "patient_id": patient_id, "name": patient_info['name'], "room": patient_info['room'], "bed": patient_info['bed'], "current_mews": current_score, "reason": reason })
        return {"critical_patients": critical_list}

    def get_patient_vitals_trend(self, params: dict) -> dict:
        """
        Generates time-series vitals data for a single patient over a duration.
        """
        self.logger.log(f"TOOL: Getting vitals trend with params: {params}")
        patient = self._find_patient(params)
        if not patient:
            return {"error": "Patient not found."}

        duration_hours = params.get("duration_hours", 24)
        vitals_trend = []
        now = int(time.time())
        for i in range(duration_hours):
            vitals_trend.append({
                "timestamp": now - i * 3600,
                "hr": random.randint(65, 85),
                "rr": random.randint(14, 20),
                "bp": f"{random.randint(110, 130)}/{random.randint(70, 85)}",
                "temperature": round(random.uniform(97.5, 99.0), 1),
                "spo2": random.randint(96, 99),
                "ews_score": random.randint(1, 4)
            })
        
        return {
            "patients": patient["name"],
            "patient_id": patient["id"],
            "ward_name": patient["ward"],
            "room": patient["room"],
            "bed": patient["bed"],
            "doctor_name": patient["doctor"],
            "vitals": vitals_trend
        }


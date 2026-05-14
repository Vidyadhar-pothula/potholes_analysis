class PhysicsSeverityEngine:
    """
    Replaces static thresholding with dynamic, weighted multi-factor physics severity.
    """
    def __init__(self):
        # Heuristic calibration weights for road destruction index
        self.w_area = 0.35
        self.w_depth = 0.40
        self.w_volume = 0.15
        self.w_edge = 0.10
        
    def calculate_severity(self, area_m2, max_depth_cm, volume_cm3, edge_sharpness):
        """
        Returns a discrete structural risk Priority Score (0-100) and Label.
        """
        # Normalize inputs against critical failure bounds
        norm_area = min(area_m2 / 2.0, 1.0)       # Caps at 2 square meters
        norm_depth = min(max_depth_cm / 20.0, 1.0) # Caps at 20 centimeters deep
        norm_vol = min(volume_cm3 / 20000.0, 1.0)  # Caps at 20 Liters missing mass
        norm_edge = min(edge_sharpness / 50.0, 1.0)
        
        score = (
            self.w_area * norm_area +
            self.w_depth * norm_depth +
            self.w_volume * norm_vol +
            self.w_edge * norm_edge
        ) * 100.0
        
        priority = int(min(score, 100.0))
        
        if priority < 30:
            category = "Low"
        elif priority < 60:
            category = "Medium"
        elif priority < 85:
            category = "High"
        else:
            category = "Critical"
            
        return priority, category

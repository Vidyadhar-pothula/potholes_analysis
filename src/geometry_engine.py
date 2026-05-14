import numpy as np
import cv2

class GeometryEngine:
    """
    Mathematical physics engine calculating real-world geometry from metric depth maps.
    """
    def __init__(self, pixel_to_m2_ratio=0.0005):
        # Standard calibration ratio for static camera
        self.pixel_to_m2_ratio = pixel_to_m2_ratio
        
    def compute_volume(self, metric_depth_map, mask_bin):
        """
        Calculates physical pothole volume via Riemann sums over pixel depths.
        metric_depth_map: Depth map in cm
        mask_bin: binary pothole mask
        """
        active_depth = metric_depth_map[mask_bin == 1]
        if active_depth.size == 0:
            return 0.0
            
        # Convert m^2 pixel area to cm^2 (0.0005 m^2 = 5 cm^2)
        area_cm2_per_pixel = self.pixel_to_m2_ratio * 10000 
        
        # Integrate (Volume = Area * Depth)
        volume_cm3 = np.sum(active_depth * area_cm2_per_pixel)
        return float(volume_cm3)
        
    def compute_edge_sharpness(self, mask_bin, metric_depth_map):
        """
        Calculates morphological edge variance to determine structural risk to tires.
        """
        kernel = np.ones((5,5), np.uint8)
        boundary = cv2.morphologyEx(mask_bin.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        
        edge_depths = metric_depth_map[boundary == 1]
        if edge_depths.size == 0:
            return 0.0
            
        sharpness = np.var(edge_depths)
        return float(sharpness)

import json
import os

class GISExporter:
    """
    Smart City Integration Module.
    Aggregates geospatial data from tracked potholes to generate 
    GeoJSON mappings and risk heatmaps for civic databases.
    """
    def __init__(self, output_dir="outputs/gis"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.features = []

    def log_anomaly(self, pothole_id, lat, lon, severity, volume, priority_score, image_ref=None):
        """
        Logs a specific tracked pothole with GPS EXIF metadata.
        """
        # Formulate standard GeoJSON Feature
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]  # GeoJSON expects [Longitude, Latitude]
            },
            "properties": {
                "id": pothole_id,
                "severity": severity,
                "repair_priority": priority_score,
                "volume_cm3": round(volume, 2),
                "image_reference": image_ref if image_ref else "None"
            }
        }
        self.features.append(feature)
        print(f"GIS Engine logged Anomaly {pothole_id} at [{lat}, {lon}] | Priority: {priority_score}")

    def generate_geojson(self, filename="pothole_heatmap.geojson"):
        """
        Outputs all logged tracking data into an interactive Mapbox/Leaflet compatible GeoJSON file.
        """
        feature_collection = {
            "type": "FeatureCollection",
            "features": self.features
        }
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(feature_collection, f, indent=4)
            
        print(f"\n[GIS] Successfully exported {len(self.features)} potholes to {filepath}")
        return filepath
        
    def generate_csv_report(self, filename="civic_report.csv"):
        """
        Exports flat tabular data specifically formatted for local government databases.
        """
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write("Pothole_ID,Latitude,Longitude,Severity_Level,Priority_Score_100,Volume_cm3\n")
            for feat in self.features:
                prop = feat["properties"]
                coord = feat["geometry"]["coordinates"]
                f.write(f"{prop['id']},{coord[1]},{coord[0]},{prop['severity']},{prop['repair_priority']},{prop['volume_cm3']}\n")
        
        print(f"[GIS] Successfully exported Civic Data CSV to {filepath}")
        return filepath

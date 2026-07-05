"""
智慧道路高精地图重建Demo
"""
from Traffic_SmartRoad.hardware_adapt.vehicle_lidar_parser import VehicleLidarParser
from Traffic_SmartRoad.noise_optimize.dynamic_traffic_denoise import DynamicTrafficDenoise
from Traffic_SmartRoad.scene_constraint.road_flatness_constraint import RoadFlatnessConstraint
from Common_LSG.slice_rebuild import SectionRebuilder

def road_hdmap_pipeline(pcd_path, output_path):
    parser = VehicleLidarParser()
    road_data = parser.load_lidar_pointcloud(pcd_path)
    
    denoiser = DynamicTrafficDenoise()
    cleaned_stack = denoiser.dynamic_vehicle_remove(road_data.section_stack)
    
    constraint = RoadFlatnessConstraint()
    constrained_stack = [constraint.road_surface_constrain(s) for s in cleaned_stack]
    
    rebuilder = SectionRebuilder()
    hd_map = rebuilder.non_orthogonal_rebuild(constrained_stack, road_data.spacing)
    
    print("道路高精地图重建完成")
    return hd_map

if __name__ == "__main__":
    road_hdmap_pipeline("./road.pcd", "./output/road_hdmap.stl")
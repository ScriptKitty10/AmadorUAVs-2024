import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point, LineString



# Sets up the 25 ft boundary from the edge of the geofence
def create_inward_offset(geofence_coords, offset_distance, inner_offset_distance):
    try:
        feet_to_degrees = 1 / 364000.0
        offset_distance_deg = offset_distance * feet_to_degrees
        inner_offset_distance_deg = inner_offset_distance * feet_to_degrees

        geofence_coords_lonlat = [(lon, lat) for lat, lon in geofence_coords]

        original_polygon = Polygon(geofence_coords_lonlat)

        # First inward offset
        inward_offset_polygon = original_polygon.buffer(-offset_distance_deg)
        if inward_offset_polygon.is_empty:
            raise ValueError("Error: Inward offset resulted in an empty polygon. Adjust the offset distance.")

        # Second inward offset (helps with keeping points in)
        inner_inward_offset_polygon = original_polygon.buffer(-offset_distance_deg - inner_offset_distance_deg)
        if inner_inward_offset_polygon.is_empty:
            raise ValueError("Error: Inner inward offset resulted in an empty polygon. Adjust the offset distance.")

        return original_polygon, inward_offset_polygon, inner_inward_offset_polygon
    except Exception as e:
        print(f"Error creating inward offset: {e}")
        raise


# For plotting the geofence and new points (optional)
def plot_polygons(original_polygon, inward_offset_polygon, inner_inward_offset_polygon, flight_plan_coords, fixed_flight_plan):
    try:
        original_coords = np.array(original_polygon.exterior.coords)
        offset_coords = np.array(inward_offset_polygon.exterior.coords)
        inner_offset_coords = np.array(inner_inward_offset_polygon.exterior.coords)
        flight_plan_coords_lonlat = np.array([(lon, lat) for lat, lon in flight_plan_coords])
        fixed_coords = np.array([(lon, lat) for lat, lon in fixed_flight_plan])

        plt.figure(figsize=(10, 10))
        plt.plot(original_coords[:, 0], original_coords[:, 1], 'b-', label='Original Geofence')
        plt.fill(original_coords[:, 0], original_coords[:, 1], color='blue', alpha=0.1)
        plt.plot(offset_coords[:, 0], offset_coords[:, 1], 'r-', label='Inward Offset Geofence')
        plt.fill(offset_coords[:, 0], offset_coords[:, 1], color='red', alpha=0.1)
        plt.plot(inner_offset_coords[:, 0], inner_offset_coords[:, 1], 'orange', linestyle='--', label='Inner Inward Offset Geofence')
        plt.fill(inner_offset_coords[:, 0], inner_offset_coords[:, 1], color='orange', alpha=0.1)
        plt.plot(flight_plan_coords_lonlat[:, 0], flight_plan_coords_lonlat[:, 1], 'g--', marker='o', label='Original Flight Plan', markersize=5)
        plt.plot(fixed_coords[:, 0], fixed_coords[:, 1], 'm-', marker='x', label='Fixed Flight Plan', markersize=5)

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Original and Inward Offset Geofences with Flight Plans')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error plotting polygons: {e}")
        raise


#  Helps with finding best path 
def get_shortest_boundary_path(boundary_line, start_distance, end_distance, num_points=100): # Can choose number of points
    try:
        total_length = boundary_line.length
        forward_distance = (end_distance - start_distance) % total_length
        reverse_distance = (start_distance - end_distance) % total_length

        if forward_distance <= reverse_distance:
            if start_distance <= end_distance:
                distances = np.linspace(start_distance, end_distance, num_points)
            else:
                distances1 = np.linspace(start_distance, total_length, num_points//2)
                distances2 = np.linspace(0, end_distance, num_points//2)
                distances = np.concatenate([distances1, distances2])
        else:
            if start_distance >= end_distance:
                distances = np.linspace(start_distance, end_distance, num_points)
            else:
                distances1 = np.linspace(start_distance, 0, num_points//2)
                distances2 = np.linspace(total_length, end_distance, num_points//2)
                distances = np.concatenate([distances1, distances2])

        distances = distances % total_length
        points = [boundary_line.interpolate(distance) for distance in distances]
        coords = [(p.x, p.y) for p in points]

        return coords
    except Exception as e:
        print(f"Error getting shortest boundary path: {e}")
        raise


# The main flight plan fixing function
def fix_flight_plan(flight_plan_coords, inner_inward_offset_polygon):
    try:
        fixed_flight_plan = []
        boundary_line = inner_inward_offset_polygon.exterior
        flight_plan_coords_lonlat = [(lon, lat) for lat, lon in flight_plan_coords]

        for i in range(len(flight_plan_coords_lonlat) - 1): # Checks if point is inside boundary
            start_point = Point(flight_plan_coords_lonlat[i])
            end_point = Point(flight_plan_coords_lonlat[i + 1])

            if not inner_inward_offset_polygon.contains(start_point):
                start_point = boundary_line.interpolate(boundary_line.project(start_point))

            if not inner_inward_offset_polygon.contains(end_point):
                end_point = boundary_line.interpolate(boundary_line.project(end_point))

            line = LineString([start_point, end_point])

            if inner_inward_offset_polygon.contains(line):
                if not fixed_flight_plan or fixed_flight_plan[-1] != (start_point.y, start_point.x):
                    fixed_flight_plan.append((start_point.y, start_point.x))
                if i == len(flight_plan_coords_lonlat) - 2:
                    fixed_flight_plan.append((end_point.y, end_point.x))
            else:
                if not fixed_flight_plan or fixed_flight_plan[-1] != (start_point.y, start_point.x):
                    fixed_flight_plan.append((start_point.y, start_point.x))

                start_distance = boundary_line.project(start_point)
                end_distance = boundary_line.project(end_point)

                # Checks best path (then appends best points)

                boundary_path_coords = get_shortest_boundary_path(boundary_line, start_distance, end_distance, num_points=4)

                if boundary_path_coords and boundary_path_coords[0] == (start_point.x, start_point.y):
                    boundary_path_coords = boundary_path_coords[1:]

                if i != len(flight_plan_coords_lonlat) - 2 and boundary_path_coords and boundary_path_coords[-1] == (end_point.x, end_point.y):
                    boundary_path_coords = boundary_path_coords[:-1]

                for coord in boundary_path_coords:
                    fixed_flight_plan.append((coord[1], coord[0]))

                if i == len(flight_plan_coords_lonlat) - 2:
                    fixed_flight_plan.append((end_point.y, end_point.x))

        return fixed_flight_plan
    except Exception as e:
        print(f"Error fixing flight plan: {e}")
        raise


# Generates the navigate.plan file (for uploading in QGroundControl)
def generate_plan_file(fixed_flight_plan, geofence_coords, filename="navigate.plan"):
    try:
        altitude_ft = 100
        altitude_m = altitude_ft * 0.3048
        speed_mph = 30
        speed_ms = speed_mph * 0.44704

        waypoints = [(round(lat, 7), round(lon, 7)) for lat, lon in fixed_flight_plan]

        mission_items = []
        mission_items.append({
            "autoContinue": True,
            "command": 178,
            "doJumpId": 1,
            "frame": 2,
            "params": [1, speed_ms, -1, 0, 0, 0, 0],
            "type": "SimpleItem"
        })

        for idx, (lat, lon) in enumerate(waypoints):
            item = {
                "AMSLAltAboveTerrain": altitude_m,
                "Altitude": altitude_m,
                "AltitudeMode": 1,
                "autoContinue": True,
                "command": 16,
                "doJumpId": idx + 2,
                "frame": 3,
                "params": [0, 0, 0, 0, lat, lon, altitude_m],
                "type": "SimpleItem"
            }
            mission_items.append(item)

        if waypoints:
            home_lat, home_lon = waypoints[0]
        else:
            raise ValueError("No waypoints to set as home position.")

        planned_home_position = [home_lat, home_lon, altitude_m]

        geofence_polygon = [[round(lat, 7), round(lon, 7)] for lat, lon in geofence_coords]

        plan = {
            "fileType": "Plan",
            "version": 1,
            "groundStation": "QGroundControl",
            "mission": {
                "version": 2,
                "firmwareType": 12,
                "vehicleType": 2,
                "cruiseSpeed": speed_ms,
                "hoverSpeed": speed_ms,
                "items": mission_items,
                "plannedHomePosition": planned_home_position
            },
            "geoFence": {
                "version": 2,
                "polygons": [
                    {
                        "version": 1,
                        "inclusion": True,
                        "polygon": geofence_polygon
                    }
                ],
                "circles": []
            },
            "rallyPoints": {
                "version": 2,
                "points": []
            }
        }

        with open(filename, 'w') as f:
            json.dump(plan, f, indent=4)

    except Exception as e:
        print(f"Error generating plan file: {e}")
        raise


# Main function to run everything (with error handling)
def main():
    parser = argparse.ArgumentParser(description="Fixes the flight plan entered by user")
    parser.add_argument("input_file", help="Path to the input text file containing geofence and flight plan coordinates.")
    
    args = parser.parse_args()

    try:
        coordinates = []
        flight_plan_coords = []

        with open(args.input_file, "r") as f:
            lines = f.readlines()
            N, M = map(int, lines[0].split())
            for line in lines[1:N+1]:
                lat, lon = map(float, line.strip().split())
                coordinates.append((lat, lon))
            for line in lines[N+1:N+1+M]:
                lat, lon = map(float, line.strip().split())
                flight_plan_coords.append((lat, lon))

        geofence_coords = coordinates
        offset_distance = 25
        inner_offset_distance = 23

        original_polygon, offset_polygon, inner_offset_polygon = create_inward_offset(geofence_coords, offset_distance, inner_offset_distance)
        fixed_flight_plan = fix_flight_plan(flight_plan_coords, inner_offset_polygon)

        generate_plan_file(fixed_flight_plan, geofence_coords, filename="navigate.plan")
        plot_polygons(original_polygon, offset_polygon, inner_offset_polygon, flight_plan_coords, fixed_flight_plan)

        print("Original Flight Plan:")
        print(flight_plan_coords)
        print("\nFixed Flight Plan:")
        print(fixed_flight_plan)

    except FileNotFoundError:
        print(f"Error: The file '{args.input_file}' was not found.")
    except ValueError as ve:
        print(f"Error parsing input file: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()

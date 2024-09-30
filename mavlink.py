import asyncio
from mavsdk import System
from mavsdk.offboard import PositionNedYaw

# Connects to the drone
async def connect_drone():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Drone connected!")
            break

    return drone

async def monitor_mission(drone):
    print("Waiting for mission to start...")
    async for mission_progress in drone.mission.mission_progress():
        print(f"Mission progress: "
              f"{mission_progress.current}/"
              f"{mission_progress.total}")
        if mission_progress.current == mission_progress.total:
            print("Mission completed!")
            return True 
        await asyncio.sleep(1)
    return False

async def switch_to_offboard(drone):
    print("-- Setting initial offboard setpoint")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))

    print("-- Starting offboard mode")
    try:
        await drone.offboard.start()
        print("Offboard mode started successfully!")
        return True
    except Exception as e:
        print(f"Failed to start offboard mode: {e}")
        await drone.action.disarm()
        return False

async def fly_to_ned(drone, north, east, down, yaw):
    print(f"-- Flying to NED position: N{north}, E{east}, D{down}")
    await drone.offboard.set_position_ned(PositionNedYaw(north, east, down, yaw))
    await asyncio.sleep(10)  # optional sleep

# Drop the payload
async def drop_payload(drone):
    print("-- Opening Actuator 1 to drop payload")
    await drone.action.actuator_control([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    await asyncio.sleep(2)  # also optional

    print("-- Closing Actuator 1")
    await drone.action.actuator_control([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


# Checks telemetry health (important for next function)
async def check_telemetry(drone):
    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            return True
        return False

# This runs in an async function in the background
async def background_telemetry_check(drone):
    while True:  # Checks constantly
        print("-- Checking health of telemetry")
        telemetry_ok = await check_telemetry(drone)

        if not telemetry_ok: # First check, waits 30 seconds
            print("-- Telemetry lost! Waiting for 30 seconds.")
            await asyncio.sleep(30)

            telemetry_ok = await check_telemetry(drone)
            if not telemetry_ok: # 2nd check, waits 3 minutes
                print("-- Telemetry still lost! Switching to RTL mode.")
                await drone.action.return_to_launch()

                print("-- Waiting for 3 minutes in RTL mode.")
                await asyncio.sleep(180)

                telemetry_ok = await check_telemetry(drone) 
                if not telemetry_ok: # 3rd check, kills motors
                    print("-- Telemetry still lost! Killing all motor operations.")
                    await drone.action.kill()
                    return  

        print("-- Telemetry OK. Continuing background checks.")
        await asyncio.sleep(10) # checks every 10 seconds

async def main():
    drone = await connect_drone()

    # Asynchronous checking
    asyncio.create_task(background_telemetry_check(drone))

    mission_complete = await monitor_mission(drone)
    if mission_complete:

        offboard_started = await switch_to_offboard(drone)
        if offboard_started:

            await fly_to_ned(drone, north=20.0, east=10.0, down=-5.0, yaw=90.0)


            await drop_payload(drone)


            print("-- Stopping offboard mode")
            await drone.offboard.stop()

            print("-- Landing")
            await drone.action.land()

    print("Mission, offboard operation, and payload drop completed. Stopping script.")


# Run script!
if __name__ == "__main__":
    asyncio.run(main())

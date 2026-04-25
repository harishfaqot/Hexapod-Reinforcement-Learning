import pybullet as p
import pybullet_data
import numpy as np
import math

def add_terrain():
    # Heightfield parameters (smaller terrain)
    size = 10  # 64x64 grid
    height_scale = np.random.uniform(0.1, 0.2)  # Controls terrain roughness

    # Generate small random terrain
    terrain_shape = np.random.uniform(low=-1, high=1, size=(size, size))
    terrain_shape = (terrain_shape / np.max(np.abs(terrain_shape))).flatten()

    # Create heightfield
    terrain_id = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[0.5, 0.5, height_scale],  # Smaller scale
        heightfieldTextureScaling=(size - 1) / 2,
        heightfieldData=terrain_shape,
        numHeightfieldRows=size,
        numHeightfieldColumns=size
    )

    # Create terrain body
    terrain_body = p.createMultiBody(0, terrain_id)
    p.changeVisualShape(terrain_body, -1, rgbaColor=[0.3, 0.3, 0.7, 1])  # Greenish terrain

    return terrain_body

def add_field():
    terrain_start_pos = [1.97, 0, 0]
    terrain_orientation = p.getQuaternionFromEuler([0, 0, 0])
    terrain_id = p.loadURDF("models/lapangan.urdf", terrain_start_pos, terrain_orientation, useFixedBase=True)
    p.changeDynamics(terrain_id, -1, lateralFriction=5)

    terrain_start_pos = [0.98, 0, 0]
    terrain_orientation = p.getQuaternionFromEuler([0, 0, 0])
    terrain1_id = p.loadURDF("models/lapangan.urdf", terrain_start_pos, terrain_orientation, useFixedBase=True)
    p.changeDynamics(terrain1_id, -1, lateralFriction=5)

    terrain_start_pos = [0, 0, (0.5 * math.sin(math.radians(10))) + 0.005]
    terrain_orientation = p.getQuaternionFromEuler([0, math.radians(10), 0])
    terrain2_id = p.loadURDF("models/lapangan.urdf", terrain_start_pos, terrain_orientation, useFixedBase=True)
    p.changeDynamics(terrain2_id, -1, lateralFriction=5)

    terrain_start_pos = [-0.98, 0, 0.178]
    terrain_orientation = p.getQuaternionFromEuler([0, 0, 0])
    terrain3_id = p.loadURDF("models/lapangan.urdf", terrain_start_pos, terrain_orientation, useFixedBase=True)
    p.changeDynamics(terrain3_id, -1, lateralFriction=5)

    terrain_start_pos = [-1.97, 0, 0.178/2]
    terrain_orientation = p.getQuaternionFromEuler([0, math.radians(-10), 0])
    terrain4_id = p.loadURDF("models/lapangan.urdf", terrain_start_pos, terrain_orientation, useFixedBase=True)
    p.changeDynamics(terrain4_id, -1, lateralFriction=5)

    terrain_start_pos = [-2.96, 0, 0]
    terrain_orientation = p.getQuaternionFromEuler([0, 0, 0])
    terrain5_id = p.loadURDF("models/lapangan.urdf", terrain_start_pos, terrain_orientation, useFixedBase=True)
    p.changeDynamics(terrain5_id, -1, lateralFriction=5)

    terrain_start_pos = [-2.96 - 0.5, 0, 0]
    terrain_orientation = p.getQuaternionFromEuler([0, 0, math.radians(180)])
    terrain6_id = p.loadURDF("models/tangga.urdf", terrain_start_pos, terrain_orientation, useFixedBase=True)
    p.changeDynamics(terrain6_id, -1, lateralFriction=5)

    terrain_start_pos = [-4.85, 0, 0.5]
    terrain_orientation = p.getQuaternionFromEuler([0, 0, 0])
    terrain7_id = p.loadURDF("models/lapangan.urdf", terrain_start_pos, terrain_orientation, useFixedBase=True)
    p.changeDynamics(terrain7_id, -1, lateralFriction=5)

    terrain_start_pos = [-5.85, 0, 0.5]
    terrain_orientation = p.getQuaternionFromEuler([0, 0, 0])
    terrain8_id = p.loadURDF("models/lapangan.urdf", terrain_start_pos, terrain_orientation, useFixedBase=True)
    p.changeDynamics(terrain8_id, -1, lateralFriction=5)

    # Add simple static box obstacles to make the path harder.
    obstacle_specs = [
        # center blocks
        {"half": [0.10, 0.10, 0.12], "pos": [1.0, 0.2, 0.12]},
        {"half": [0.12, 0.10, 0.14], "pos": [0.0, 0.5, 0.14]},
        {"half": [0.10, 0.12, 0.16], "pos": [-1.0, -0.5, 0.16]},
        {"half": [0.10, 0.10, 0.12], "pos": [-2.0, -0.5, 0.12]},
        {"half": [0.12, 0.10, 0.14], "pos": [-3.0, 0.0, 0.14]},
        {"half": [0.10, 0.12, 0.16], "pos": [-4.0, -0.00, 0.16]},
    ]

    for obs in obstacle_specs:
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=obs["half"])
        vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=obs["half"], rgbaColor=[0.45, 0.30, 0.25, 1])
        box_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=obs["pos"],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        )
        p.changeDynamics(box_id, -1, lateralFriction=3.5, rollingFriction=0.02, spinningFriction=0.02)

if __name__ == '__main__':
    # Initialize physics engine
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    # Load terrain
    terrain = add_field()

    # Run simulation loop
    while True:
        p.stepSimulation()

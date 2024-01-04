import math

# tensorflow doesnt support gymnasium
# import gymnasium as gym
# from gymnasium import spaces
import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data


OBJ_STATE_NUM = 13
MAX_TORQUE = 5


class BasicSkateboardEnv(gym.Env):
    def __init__(self, mode="direct", ground_path="plane.urdf", gravity=9.8):
        self.render_image = None
        self._ground_path = ground_path
        self._gravity = gravity
        if mode == "direct":
            self.client = p.connect(p.DIRECT)
        elif mode == "gui":
            self.client = p.connect(p.GUI)
        else:
            raise ValueError("mode must be either 'direct' or 'gui'")

        self._init_env()

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(OBJ_STATE_NUM + len(self._agent_joint_index) * 8,),
                ),
                "skateboard": spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(OBJ_STATE_NUM + len(self._sb_joint_index) * 2,),
                ),
            }
        )

        # action space for position control
        # self.action_space = spaces.Box(low=-math.pi, high=math.pi,
        #                                shape=(len(self._agent_joint_index),))
        self.action_space = spaces.Box(
            low=-MAX_TORQUE, high=MAX_TORQUE, shape=(len(self._agent_joint_index),)
        )

    def _init_env(self) -> None:
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -self._gravity)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setPhysicsEngineParameter(numSolverIterations=5)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240.0)
        p.setPhysicsEngineParameter(numSubSteps=1)

        self._agent_location = self._agent_init_location()
        self._agent_orientation = self._agent_init_orientation()
        self._agent_velocities = [0, 0, 0]
        self._skateboard_location = self._init_skateboard_location()
        self._skateboard_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self._skateboard_velocities = [0, 0, 0]
        self._ground_id = self._load_ground(self._ground_path)
        (
            self._agent_id,
            self._agent_part_map,
            self._agent_joint_index,
        ) = self._load_agent(self._agent_location, self._agent_orientation)
        (
            self._skateboard_id,
            self._skatboard_part_map,
            self._sb_joint_index,
        ) = self._load_skateboard(
            self._skateboard_location, self._skateboard_orientation
        )
        self._agent_joint_info = np.zeros((5*len(self._sb_joint_index)))
        self._skateboard_joint_info = np.zeros(2*len(self._sb_joint_index))

    def _agent_init_location(self) -> list:
        return [0, 0, 0.1]  # np.random.uniform(0.5, 0.6, (1, 3)).tolist()[0]

    def _agent_init_orientation(self) -> list:
        # return p.getQuaternionFromEuler(list(np.random.uniform(-math.pi/3, math.pi/3, size=3)))
        return p.getQuaternionFromEuler([0, 0, 0])

    def _init_skateboard_location(self) -> list:
        return [5, 5, 5]  # np.random.uniform(0.2, 0.4, (1, 3)).tolist()[0]

    def _load_ground(self, ground_path) -> None:
        ground = p.loadURDF(ground_path)
        for link_index in range(p.getNumJoints(ground) + 1):
            p.changeDynamics(ground, link_index, lateralFriction=0.75,
                             rollingFriction=0.012, restitution=0)
        return ground

    def _load_agent(self, humanoid_start_pos, humanoid_start_orientation) -> None:
        print("Loading agent...")
        humanoid = p.loadURDF(
            "urdf_files/humanoid.urdf",
            humanoid_start_pos,
            humanoid_start_orientation,
            globalScaling=0.1,
        )

        humanoidNameToId = {}
        info = p.getBodyInfo(humanoid)
        humanoidNameToId[info[0].decode("utf-8")] = -1
        joint_indexs = []
        for i in range(p.getNumJoints(humanoid)):
            joint_info = p.getJointInfo(humanoid, i)
            humanoidNameToId[joint_info[12].decode("UTF-8")] = i
            humanoidNameToId[joint_info[1].decode("UTF-8")] = i
            if joint_info[2] == p.JOINT_REVOLUTE:
                joint_indexs.append(i)

        for link_index in range(p.getNumJoints(humanoid) + 2):
            humanoid_restitution = 0
            # humanoid_contact_damping = 1000
            # humanoid_contact_stiffness = 10000
            p.changeDynamics(humanoid, link_index - 1, restitution=humanoid_restitution)
            # contactDamping=humanoid_contact_damping,
            # contactStiffness=humanoid_contact_stiffness)

        for joint_index in range(p.getNumJoints(humanoid)):
            p.enableJointForceTorqueSensor(humanoid, joint_index, enableSensor=True)

        print("finished loading agent")
        return humanoid, humanoidNameToId, joint_indexs

    def _load_skateboard(self, sb_start_pos, sb_start_orientation) -> None:
        print("Loading skateboard...")
        skateboard = p.loadURDF(
            "urdf_files/skateboard.urdf",
            sb_start_pos,
            sb_start_orientation,
            globalScaling=0.12,
            useFixedBase=False,
        )

        skateboardNameToId = {}
        joint_indexs = []
        for i in range(p.getNumJoints(skateboard)):
            joint_info = p.getJointInfo(skateboard, i)
            skateboardNameToId[joint_info[12].decode("UTF-8")] = i
            skateboardNameToId[joint_info[1].decode("UTF-8")] = i
            if joint_info[2] == p.JOINT_REVOLUTE:
                joint_indexs.append(i)

        deck = -1
        deck_curve1 = skateboardNameToId["deck_curve1"]
        deck_curve2 = skateboardNameToId["deck_curve2"]
        steering = skateboardNameToId["steering"]
        front_left_wheel = skateboardNameToId["front_left_wheel"]
        front_right_wheel = skateboardNameToId["front_right_wheel"]
        back_left_wheel = skateboardNameToId["back_left_wheel"]
        back_right_wheel = skateboardNameToId["back_right_wheel"]

        # change dynamic settings:
        sb_board_restitution = 0
        sb_board_lateral_friction = 1
        sb_board_contact_damping = 100
        sb_board_contact_stiffness = 10000

        p.changeDynamics(
            skateboard,
            deck,
            lateralFriction=sb_board_lateral_friction,
            restitution=sb_board_restitution,
            contactDamping=sb_board_contact_damping,
            contactStiffness=sb_board_contact_stiffness,
        )
        p.changeDynamics(
            skateboard,
            deck_curve1,
            lateralFriction=sb_board_lateral_friction,
            restitution=sb_board_restitution,
            contactDamping=sb_board_contact_damping,
            contactStiffness=sb_board_contact_stiffness,
        )
        p.changeDynamics(
            skateboard,
            deck_curve2,
            lateralFriction=sb_board_lateral_friction,
            restitution=sb_board_restitution,
            contactDamping=sb_board_contact_damping,
            contactStiffness=sb_board_contact_stiffness,
        )

        sb_wheel_lateral_friction = 0.2
        sb_wheel_restitution = 0
        sb_wheel_roll_friction = 0.2
        sb_wheel_spin_friction = 0.02
        sb_wheel_contact_damping = 12
        sb_wheel_contact_stiffness = 100000

        p.changeDynamics(
            skateboard,
            front_left_wheel,
            lateralFriction=sb_wheel_lateral_friction,
            restitution=sb_wheel_restitution,
            rollingFriction=sb_wheel_roll_friction,
            spinningFriction=sb_wheel_spin_friction,
            contactDamping=sb_wheel_contact_damping,
            contactStiffness=sb_wheel_contact_stiffness,
        )

        p.changeDynamics(
            skateboard,
            front_right_wheel,
            lateralFriction=sb_wheel_lateral_friction,
            restitution=sb_wheel_restitution,
            rollingFriction=sb_wheel_roll_friction,
            spinningFriction=sb_wheel_spin_friction,
            contactDamping=sb_wheel_contact_damping,
            contactStiffness=sb_wheel_contact_stiffness,
        )

        p.changeDynamics(
            skateboard,
            back_left_wheel,
            lateralFriction=sb_wheel_lateral_friction,
            restitution=sb_wheel_restitution,
            rollingFriction=sb_wheel_roll_friction,
            spinningFriction=sb_wheel_spin_friction,
            contactDamping=sb_wheel_contact_damping,
            contactStiffness=sb_wheel_contact_stiffness,
        )

        p.changeDynamics(
            skateboard,
            back_right_wheel,
            lateralFriction=sb_wheel_lateral_friction,
            restitution=sb_wheel_restitution,
            rollingFriction=sb_wheel_roll_friction,
            spinningFriction=sb_wheel_spin_friction,
            contactDamping=sb_wheel_contact_damping,
            contactStiffness=sb_wheel_contact_stiffness,
        )

        sb_steering_contact_damping = 10
        sb_steering_contact_stiffness = 100
        sb_steering_restitution = 1
        p.changeDynamics(
            skateboard,
            steering,
            contactDamping=sb_steering_contact_damping,
            contactStiffness=sb_steering_contact_stiffness,
            restitution=sb_steering_restitution,
        )

        for joint_index in range(p.getNumJoints(skateboard)):
            p.enableJointForceTorqueSensor(skateboard, joint_index, enableSensor=True)

        print("finished loading skateboard")
        return skateboard, skateboardNameToId, joint_indexs

    def _apply_clamp_force(self, obj_id, index) -> None:
        for joint_index in index:
            joint_info = p.getJointInfo(obj_id, joint_index)
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            joint_state = p.getJointState(obj_id, joint_index)
            pos = joint_state[0]
            if pos < lower_limit:
                pos = lower_limit
            elif pos > upper_limit:
                pos = upper_limit
            if pos != joint_state[0]:
                p.setJointMotorControl2(
                    obj_id, joint_index, p.POSITION_CONTROL, 
                    targetVelocity=0, targetPosition=pos, force=0)
                p.setJointMotorControl2(obj_id, joint_index, p.TORQUE_CONTROL, force=0)

    def _get_obs(self) -> dict:
        # agent
        # location(3) + orientation(4) + linear velocity(3) + angular velocity(3) +
        # joint states(8N for agent, 2N for skateboard)
        return {
            "agent": np.array(
                list(self._agent_location) +
                list(self._agent_orientation) +
                self._agent_velocities +
                self._agent_joint_info,
                dtype=np.float32
            ),
            "skateboard": np.hstack(
                list(self._skateboard_location) +
                list(self._skateboard_orientation) +
                self._skateboard_velocities +
                self._skateboard_joint_info,
                dtype=np.float32
            ),
        }

    def get_contact_points(self, link_name=None, target_id="ground"):
        index = None
        if target_id == "ground":
            index = self._ground_id
        elif target_id == "skateboard":
            index = self._skateboard_id
        else:
            raise ValueError("target_id must be either 'ground' or 'skateboard'")
        if link_name == None:
            points = p.getContactPoints(self._agent_id, index)
            return p.getContactPoints(self._agent_id, index)
        else:
            return p.getContactPoints(
                self._agent_id, index, linkIndexA=self._agent_part_map[link_name])

    def get_link_pos(self, link_name) -> tuple:
        return p.getLinkState(self._agent_id, self._agent_part_map[link_name])[0]

    def get_link_orientation(self, link_name) -> tuple:
        return p.getLinkState(self._agent_id, self._agent_part_map[link_name])[1]

    def get_position(self) -> tuple:
        return self._agent_location

    def get_orientation(self, link_name=None) -> tuple:
        if link_name==None:
            return self._agent_orientation
        else:
            return p.getLinkState(self._agent_id, self._agent_part_map[link_name])[1]

    def get_joint_states(self) -> list:
        return p.getJointStates(self._agent_id, self._agent_joint_index)

    def _update_info(self) -> None:
        self._agent_location, self._agent_orientation \
            = p.getBasePositionAndOrientation(self._agent_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(self._agent_id)
        self._agent_velocities = list(linear_velocity + angular_velocity)
        joint_states = p.getJointStates(self._agent_id, self._agent_joint_index)
        joint_info = []
        # for index in range(len(self._agent_joint_index)):
        #     # normal reaction force
        #     contact_point_info = p.getContactPoints(
        #         self._agent_id, self._ground_id, linkIndexA=index
        #     )
        #     normal_sum = np.array([0, 0, 0])
        #     if contact_point_info is not None:
        #         for contact_point in contact_point_info:
        #             normal_sum = np.add(normal_sum, contact_point[7])
        #     joint_info += list(joint_states[index][0:2]) + list(normal_sum)
        for joint_state in joint_states:
            joint_info += list(joint_state[0:2]) + list(joint_state[2])
        self._agent_joint_info = joint_info

        self._skateboard_location, self._skateboard_orientation \
            = p.getBasePositionAndOrientation(self._skateboard_id)
        linear_velocity, angular_velocity = p.getBaseVelocity(self._skateboard_id)
        self._skateboard_velocities = list(linear_velocity + angular_velocity)
        joint_states = p.getJointStates(self._agent_id, self._sb_joint_index)
        self._skateboard_joint_info = \
            [joint_state[i] for joint_state in joint_states for i in range(2)]

    def reset(self, seed=None, options=None) -> dict:
        super().reset(seed=seed)
        print("resetting...")
        p.resetSimulation()
        self._init_env()
        self._update_info()
        observation = self._get_obs()
        return observation

    def step(self, action) -> tuple:
        # print("action: ", action)
        zeros = np.zeros_like(action)
        p.setJointMotorControlArray(
            self._agent_id, self._agent_joint_index, p.VELOCITY_CONTROL, forces=zeros
        )
        p.setJointMotorControlArray(
            self._agent_id, self._agent_joint_index, p.TORQUE_CONTROL, forces=action
        )
        self._apply_clamp_force(self._skateboard_id, self._sb_joint_index)
        self._apply_clamp_force(self._agent_id, self._agent_joint_index)
        p.stepSimulation()

        self._update_info()

        observation = self._get_obs()
        return observation, 0, True, False

    def close(self):
        p.disconnect()
        super().close()

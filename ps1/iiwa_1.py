# iiwa problems for pset 1

import numpy as np
from dtsystems import (
    Constant,
    HEADLESS,
    Observer,
    PController,
    PDController,
    PDController2,
    SimpleTrajectoryFollower,
    get_meshcat,
    get_positions,
    get_state,
    series_composition,
    show_meshcat,
    simulate,
    plot_log,
)
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Diagram,
    DiagramBuilder,
    LogVectorOutput,
    Meshcat,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    plot_system_graphviz,
)

from utils.drake_models import explain_model_download_error
from utils.plotting import plt

######################################################################
## Code for students to be aware of
######################################################################

IIWA14_URL = (
    "package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf"
)

Q_START = np.array([0, 1.0, 0.3, 0.7, 0, 0, 0])

# The joint-space step used in the PD-controller question: drive the arm from
# Q_START to Q_START + Q_STEP (a step on joints 1, 2 and 4).
Q_STEP = np.array([0.5, -0.3, 0, 0.4, 0, 0, 0])

PLANT_DT = 1e-4

def iiwa_s0(q0: np.ndarray) -> list[np.ndarray]:
    """
    Initial state list for simulate(): group 0 is the plant, whose discrete
    state is [q, v].
    """
    return [np.hstack([q0, np.zeros(7)])]


# Joint configurations that put the end effector at the corners of a 0.4 m
# square in the vertical plane x = 0.5, with the end-effector frame held
# axis-aligned, found (offline) by inverse kinematics -- a topic we will
# study properly later in the term.  Closed by repeating the first corner,
# like SQUARE in cartesian_2d_robot.py.  Drive the arm through them in order
# and the end effector draws the square.
SQUARE_IIWA = [
    np.array([-0.3743, 1.2271, 0.0031, 0.861, 0.0169, -0.3627, 0.3545]),
    np.array([0.2311, 1.894, 0.8932, 1.6624, -1.2178, -0.9016, 1.1868]),
    np.array([0.9917, 1.894, 0.8958, 1.6696, -1.2453, -0.9026, 0.4584]),
    np.array([0.7547, 1.1085, 0.6942, 0.8787, -1.4182, -0.6161, 0.2746]),
]
SQUARE_IIWA.append(SQUARE_IIWA[0])

######################################################################
## Code for students to study
######################################################################


def create_IIWA14_diagram(
    torques: np.ndarray = np.zeros(7), meshcat: Meshcat | None = None
) -> tuple[Diagram, MultibodyPlant]:
    """
    Build a diagram holding the welded-base iiwa14 MultibodyPlant, with a
    Constant source applying the given joint torques at the actuation input,
    a state logger named "log_plant", and a MeshcatVisualizer when meshcat is
    given.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
    parser = Parser(plant, scene_graph)
    try:
        parser.AddModelsFromUrl(IIWA14_URL)
    except RuntimeError as e:
        # The first load downloads the models; this explains the one common
        # way that fails (a space or such in the venv's path) before re-raising.
        explain_model_download_error(e)
        raise
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
    plant.Finalize()

    source = builder.AddSystem(Constant(torques))
    builder.Connect(source.get_output_port(), plant.get_actuation_input_port())

    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    logger = LogVectorOutput(plant.get_state_output_port(), builder)
    logger.set_name("log_plant")

    diagram = builder.Build()
    diagram.set_name("plant and scene_graph")
    return diagram, plant


def test_const_torque(q_initial: np.ndarray, torques: np.ndarray) -> None:
    """
    Simulate the arm from q_initial under a constant joint-torque vector.
    """
    meshcat = get_meshcat()
    diagram, plant = create_IIWA14_diagram(torques=torques, meshcat=meshcat)
    if not HEADLESS:
        plt.figure(figsize=(12, 6))
        plot_system_graphviz(diagram)
        plt.show()
    simulator = simulate(diagram, iiwa_s0(q_initial), 5.0)
    plot_log(diagram, simulator, "log_plant")
    q_final = get_positions(plant, simulator)
    print(f"   Initial joint positions: {q_initial}")
    print(f"   Final joint positions:   {q_final}")
    show_meshcat()


######################################################################
##  Code for students to write
######################################################################


def create_IIWA14_diagram_with_pcontroller(
    controller_gain: float, q_desired: np.ndarray, meshcat: Meshcat | None = None
) -> tuple[Diagram, MultibodyPlant]:
    """
    Like create_IIWA14_diagram, but the joints are driven by a proportional
    position controller: compose an Observer that keeps the positions out of
    the 14-D plant state [q, v] with a PController with the given gain and
    target q_desired, wired from the plant's state output back into its
    actuation input.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
    parser = Parser(plant, scene_graph)
    try:
        parser.AddModelsFromUrl(IIWA14_URL)
    except RuntimeError as e:
        explain_model_download_error(e)
        raise
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
    plant.Finalize()

    observer = builder.AddSystem(Observer(14, range(7)))   # [q, v] -> q
    controller = builder.AddSystem(PController(7, q_desired, controller_gain))

    builder.Connect(plant.get_state_output_port(), observer.get_input_port())
    builder.Connect(observer.get_output_port(), controller.get_input_port())
    builder.Connect(controller.get_output_port(), plant.get_actuation_input_port())

    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    logger = LogVectorOutput(plant.get_state_output_port(), builder)
    logger.set_name("log_plant")

    diagram = builder.Build()
    diagram.set_name("iiwa with P controller")
    return diagram, plant


def create_IIWA14_diagram_with_pd_controller(
    controller_gain: float,
    damping_gain: float,
    q_desired: np.ndarray,
    dt: float = 0.01,
    meshcat: Meshcat | None = None,
) -> tuple[Diagram, MultibodyPlant]:
    """
    Like create_IIWA14_diagram_with_pcontroller, but with the PDController
    defined in dtsystems.py: controller_gain, damping_gain, and dt go through
    to it.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=1e-4)
    parser = Parser(plant, scene_graph)
    try:
        parser.AddModelsFromUrl(IIWA14_URL)
    except RuntimeError as e:
        explain_model_download_error(e)
        raise
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
    plant.Finalize()

    observer = builder.AddSystem(Observer(14, range(7)))
    controller = builder.AddSystem(
        PDController(7, q_desired, controller_gain, damping_gain, dt)
    )

    builder.Connect(plant.get_state_output_port(), observer.get_input_port())
    builder.Connect(observer.get_output_port(), controller.get_input_port())
    builder.Connect(controller.get_output_port(), plant.get_actuation_input_port())

    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    logger = LogVectorOutput(plant.get_state_output_port(), builder)
    logger.set_name("log_plant")

    diagram = builder.Build()
    diagram.set_name("iiwa with PD controller")
    return diagram, plant


def create_IIWA14_diagram_with_waypoints(
    waypoints: list[np.ndarray],
    controller_gain: float = 10000,
    damping_gain: float = 3000,
    epsilon: float = 0.01,
    meshcat: Meshcat | None = None,
    ctrl_dt: float = PLANT_DT
) -> tuple[Diagram, MultibodyPlant]:
    """
    Drive the arm through a sequence of joint-space waypoints: an Observer
    keeps the positions out of the 14-D plant state, a SimpleTrajectoryFollower
    over the waypoints outputs the current target, and a PDController2 turns
    target and actual into joint torques.

    The follower only advances once the arm is within epsilon of the current
    waypoint, so this controller must be stiff enough that its gravity sag
    stays well under epsilon.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=PLANT_DT)
    parser = Parser(plant, scene_graph)
    try:
        parser.AddModelsFromUrl(IIWA14_URL)
    except RuntimeError as e:
        explain_model_download_error(e)
        raise
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))
    plant.Finalize()

    observer = builder.AddSystem(Observer(14, range(7)))
    follower = builder.AddSystem(SimpleTrajectoryFollower(waypoints, epsilon))
    controller = builder.AddSystem(
        PDController2(7, controller_gain, damping_gain, dt=ctrl_dt)
    )

    builder.Connect(plant.get_state_output_port(), observer.get_input_port())
    builder.Connect(observer.get_output_port(), follower.get_input_port())
    builder.Connect(follower.get_output_port(), controller.GetInputPort("target"))
    builder.Connect(observer.get_output_port(), controller.GetInputPort("actual"))
    builder.Connect(controller.get_output_port(), plant.get_actuation_input_port())

    if meshcat is not None:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    logger = LogVectorOutput(plant.get_state_output_port(), builder)
    logger.set_name("log_plant")

    diagram = builder.Build()
    diagram.set_name("iiwa drawing a square")
    return diagram, plant



def main1(gain: float = 100, T: float = 10.0) -> None:
    meshcat = get_meshcat()
    diagram, plant = create_IIWA14_diagram_with_pcontroller(gain, Q_START, meshcat=meshcat)
    if not HEADLESS:
        plt.figure(figsize=(12, 6))
        plot_system_graphviz(diagram)
        plt.show()
    simulator = simulate(diagram, iiwa_s0(Q_START), T)
    q_final = get_positions(plant, simulator)
    print(f"   Target joint positions: {Q_START}")
    print(f"   Final joint positions:  {q_final}")
    print(f"   Error:                  {Q_START - q_final}")
    plot_log(diagram, simulator, "log_plant")
    show_meshcat()


def main2(gain: float = 100, d_gain: float = 30,
                       ctrl_dt: float = PLANT_DT, T: float = 150.0) -> None:
    meshcat = get_meshcat()
    q_goal = Q_START + Q_STEP
    diagram, plant = create_IIWA14_diagram_with_pd_controller(
        gain, d_gain, q_goal, dt=ctrl_dt, meshcat=meshcat
    )
    if not HEADLESS:
        plt.figure(figsize=(12, 6))
        plot_system_graphviz(diagram)
        plt.show()

    # group 0 = plant [q, v]; group 1 = PD controller's "previous position"
    s0 = iiwa_s0(Q_START) + [Q_START.copy()]
    simulator = simulate(diagram, s0, T)

    q_final = get_positions(plant, simulator)
    print(f"   Goal:        {q_goal}")
    print(f"   Final:       {q_final}")
    print(f"   Final error: {q_goal - q_final}")

    log = diagram.GetSubsystemByName("log_plant").FindLog(simulator.get_context())
    t, d = log.sample_times(), log.data()
    speed = np.linalg.norm(d[7:].T, axis=1)      # joint-velocity magnitude
    tol = 1e-3
    moving = speed > tol
    if moving[-1]:
        print("   Still moving at the end of the run: increase T")
    else:
        print(f"   At rest (|v| < {tol}) after t = {t[np.nonzero(moving)[0][-1] + 1]:.2f} s")

    plot_log(diagram, simulator, "log_plant")
    show_meshcat()


def main3(T: float = 20.0, ctrl_dt: float =1e-2) -> None:
    meshcat = get_meshcat()
    diagram, plant = create_IIWA14_diagram_with_waypoints(
        SQUARE_IIWA, meshcat=meshcat, ctrl_dt=ctrl_dt
    )
    follower = diagram.GetSubsystemByName("SimpleTrajectoryFollower")
    if not HEADLESS:
        plt.figure(figsize=(12, 6))
        plot_system_graphviz(diagram)
        plt.show()

    q0 = SQUARE_IIWA[0]
    # group 0 = plant, group 1 = follower (waypoint index), group 2 = PD's previous position
    s0 = iiwa_s0(q0) + [np.array([0.0]), q0.copy()]
    simulator = simulate(diagram, s0, T)

    log = diagram.GetSubsystemByName("log_plant").FindLog(simulator.get_context())
    t, q = log.sample_times(), log.data()[:7]
    wp = SQUARE_IIWA[1]
    err = np.linalg.norm(q.T - wp, axis=1)
    last = t > t[-1] - 1.0
    print(f"   closest to waypoint 1: {err.min():.4f} at t = {t[err.argmin()]:.2f} s (need < 0.01)")
    print(f"   error over last 1 s:   mean {err[last].mean():.4f}, wiggle (std) {err[last].std():.2e}")
    print(f"   per-joint error at end: {np.round(wp - q[:, -1], 4)}")

    idx = int(get_state(follower, simulator)[0])
    print(f"   Waypoint reached: {idx} of {len(SQUARE_IIWA) - 1}")
    print(f"   Final error:      {SQUARE_IIWA[-1] - get_positions(plant, simulator)}")
    plot_log(diagram, simulator, "log_plant")
    show_meshcat()


if __name__ == "__main__":
    # test_const_torque(Q_START, np.zeros(7))     # zero torque
    # main1()
    # main2()
    for cdt in (1e-3, 1e-2):
        print(f"\n=== ctrl_dt = {cdt}")
        main3(ctrl_dt=cdt)
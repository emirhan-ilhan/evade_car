import math

import Box2D
import gym
import numpy as np
import pyglet
import torch
from Box2D.b2 import contactListener
from Box2D.b2 import fixtureDef
from Box2D.b2 import polygonShape
from gym import spaces

from DQN import DQN
from car_dynamics import Car
from utils import convert_action
from wrappers import *

pyglet.options["debug_gl"] = False
from pyglet import gl

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 1.8  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)

TRACK_DETAIL_STEP = 18 / SCALE
TRACK_TURN_RATE = 0.20
TRACK_WIDTH = 70 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.3, 0.3, 0.3]


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1:
            if "crash" in u1.__dict__:
                self.env.crash = True
            if "road_friction" in u1.__dict__:
                tile = u1
                obj = u2
        if u2:
            if "crash" in u2.__dict__:
                self.env.crash = True
            if "road_friction" in u2.__dict__:
                tile = u2
                obj = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 2000.0 / len(self.env.track)
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)


class CarRacing(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
        "video.frames_per_second": FPS,
    }

    def __init__(self, verbose=1):
        self.seed()
        self.contactListener = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = []
        self.barriers = []
        self.crash = False
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )
        self.show = False
        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        for b in self.barriers:
            self.world.DestroyBody(b)
        self.road = []
        self.barriers = []
        self.car.destroy()

    def _create_track(self):
        # CREATE CHECKPOINTS
        CHECKPOINTS = 20
        checkpoints = []

        for c in range(CHECKPOINTS):
            noise = np.random.uniform(-math.pi / CHECKPOINTS, math.pi / CHECKPOINTS)  # Add randomness so that the turns are random
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = np.random.uniform(TRACK_RAD / 2, 1.3 * TRACK_RAD)  # Randomize the turns' distances from the origin

            if c == 0:  # We do not want the starting point to be random (alpha=0)
                alpha = 0
                rad = 1.5 * TRACK_RAD
            elif c == CHECKPOINTS - 1:  # The last point is not random since it may cause problems when checking if a lap is finished
                alpha = 2 * math.pi * c / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.start_alpha = 2 * math.pi * (CHECKPOINTS - 0.5) / CHECKPOINTS  # We will know that we completed a lap when we pass this angle

        # CREATE TRACK POINTS THAT GO FROM ONE CHECKPOINT TO ANOTHER
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        # x and y are the coordinates of the track points
        # beta is the angle that defines the direction on which we will move while going to the next checkpoint

        dest_i = 0  # Index of the checkpoint to which we are trying to go right now
        laps = 0  # The number of times we go over all checkpoints. We will go more than one lap to make sure that we have enough points.
        track = []  # The list that contains the track points' data
        max_iters = 2500  # We will create track points until we exceed the max laps or we exceed the max iters
        # max_iters ensures that we do not get stuck in an infinite loop when the created track points do not progress
        visited_other_side = False  # True if the last position is below x axis, i.e. y < 0

        # Start generating track points
        for _ in range(max_iters):
            alpha = math.atan2(y, x)  # The angle of the current track point, intially zero

            # Since we will compare the current point's alpha with the next checkpoint's alpha to acquire positional information,
            # we want alpha to be in [0, 2*pi] like the checkpoints' alpha values.

            # Check alpha value to put it in desired interval and to check if a lap is completed:
            if alpha > 0 and visited_other_side:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi  # This will put alpha in [0, 2*pi] for the following operations on this iteration.

            # Now, we need to find a valid destination checkpoint to go towards to.
            dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]  # Take the dest_i'th element of checkpoints
            if alpha > dest_alpha:  # alpha > dest_alpha means that we have passed our destination point with our last position update
                final_straight = alpha > checkpoints[-1][0] and dest_i % len(checkpoints) == 0
                # After the final checkpoint our destination point will be the starting checkpoint and the destination alpha will be zero
                # Therefore current alpha will be larger than destination alpha until we reach the starting point
                # We will not update the destination point in this "final straight" interval
                if not final_straight:
                    dest_i += 1  # Assign the next checkpoint as destination
                dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]

            # We are using a positional update rule that uses beta because of two reasons:
            # 1) It allows us to do smoother turns in contrast to a method that would move directly from one checkpoint to another
            # 2) If we move with an algorithm that directly uses the next checkpoint's relative position, we would not be able to pass
            # that checkpoint (we cannot pass it if we go towards it)

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            # p1 vector shows the direction towards which we will update our track point position
            # r1 vector is orthogonal to p1 and it will be used to calculate how sharp we need to turn towards our destination

            dest_dx = dest_x - x  # dest_d points from current direction towards destination
            dest_dy = dest_y - y

            # Destination vector projected on rad (dot product of r1 and destination vector)
            proj = r1x * dest_dx + r1y * dest_dy
            # proj is a measure of dissimilarity between the moving direction that beta indicates and the one that destination vector indicates
            proj *= SCALE  # Scale the proj's value

            # Now, we will update beta. That is, we will update our moving direction.
            # TRACK_TURN_RATE determines how sharp the turns are allowed to be.
            if proj > 0.3:  # We need to turn CCW
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            elif proj < -0.3:  # We need to turn CW
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))  # If proj is too small, increase beta (but not too much)

            x += p1x * TRACK_DETAIL_STEP  # Update the track point positions
            y += p1y * TRACK_DETAIL_STEP

            track.append((alpha, beta, x, y))  # Add track point to track

            if laps > 4:  # 4 laps are enough to form a track
                break

        # FIND A CLOSED LOOP OF TRACK POINTS
        start = -1
        finish = -1
        for i in range(1, len(track) - 1):
            pass_through_start = track[i][0] < self.start_alpha and track[i + 1][0] >= self.start_alpha
            if pass_through_start and start == -1:
                start = i
            elif pass_through_start and finish == -1:
                finish = i
                break
        if start == -1 or finish == -1:
            return False

        track = track[start: finish - 1]

        # CREATE THE TILES USING THE TRACK POINTS
        last_beta = track[-1][1]
        last_perp_x = math.cos(last_beta)
        last_perp_y = math.sin(last_beta)  # Moving direction from the last track point

        # Check if start and end close enough
        start_end_connection = np.sqrt(
            np.square(last_perp_x * (track[0][2] - track[-1][2]))  # Distance between start and end in x direction
            + np.square(last_perp_y * (track[0][3] - track[-1][3]))  # Distance between start and end in y direction
        )

        if start_end_connection > TRACK_DETAIL_STEP:  # Distance between two successive track points is supposed to be TRACK_DETAIL_STEP
            return False

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]  # ith track point
            alpha2, beta2, x2, y2 = track[i - 1]  # (i-1)th track point
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices  # Specify the vertices of the base tile fixture
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)  # Create the tile object from fixture
            t.userData = t
            c = 0.01
            t.color = [ROAD_COLOR[0] + c * (i % 2), ROAD_COLOR[1] + c * (i % 2), ROAD_COLOR[2] + c * (i % 2)]
            t.road_visited = False
            t.road_friction = 1.5
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)

            # Make barriers
            if abs(beta2 - beta1) < 5:
                barr1_l = (
                    x1 - TRACK_WIDTH * 1.55 * math.cos(beta1),
                    y1 - TRACK_WIDTH * 1.55 * math.sin(beta1),
                )
                barr1_r = (
                    x1 - TRACK_WIDTH * 1.5 * math.cos(beta1),
                    y1 - TRACK_WIDTH * 1.5 * math.sin(beta1),
                )
                barr2_l = (
                    x2 - TRACK_WIDTH * 1.55 * math.cos(beta2),
                    y2 - TRACK_WIDTH * 1.55 * math.sin(beta2),
                )
                barr2_r = (
                    x2 - TRACK_WIDTH * 1.5 * math.cos(beta2),
                    y2 - TRACK_WIDTH * 1.5 * math.sin(beta2),
                )
                vertices = [barr1_l, barr1_r, barr2_r, barr2_l]
                self.fd_tile.shape.vertices = vertices  # Specify the vertices of the base tile fixture
                barrl = self.world.CreateStaticBody(fixtures=self.fd_tile)  # Create the tile object from fixture
                barrl.userData = barrl
                barrl.crash = False
                self.barriers.append(barrl)
                # barrl.color = [0.1, 0.1, 0.1]
                barrl.fixtures[0].sensor = True
                # self.road_poly.append(([barr1_l, barr1_r, barr2_r, barr2_l], barrl.color))

                barr1_l = (
                    x1 + TRACK_WIDTH * 1.55 * math.cos(beta1),
                    y1 + TRACK_WIDTH * 1.55 * math.sin(beta1),
                )
                barr1_r = (
                    x1 + TRACK_WIDTH * 1.5 * math.cos(beta1),
                    y1 + TRACK_WIDTH * 1.5 * math.sin(beta1),
                )
                barr2_l = (
                    x2 + TRACK_WIDTH * 1.55 * math.cos(beta2),
                    y2 + TRACK_WIDTH * 1.55 * math.sin(beta2),
                )
                barr2_r = (
                    x2 + TRACK_WIDTH * 1.5 * math.cos(beta2),
                    y2 + TRACK_WIDTH * 1.5 * math.sin(beta2),
                )
                vertices = [barr1_l, barr1_r, barr2_r, barr2_l]
                self.fd_tile.shape.vertices = vertices  # Specify the vertices of the base tile fixture
                barrr = self.world.CreateStaticBody(fixtures=self.fd_tile)  # Create the tile object from fixture
                barrr.userData = barrr
                barrr.crash = False
                self.barriers.append(barrr)
                # barrr.color = [0.0, 0.1, 0.0]
                barrr.fixtures[0].sensor = True
                # self.road_poly.append(([barr1_l, barr1_r, barr2_r, barr2_l], barrr.color))
        self.track = track
        return True

    def reset(self):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.crash = False
        self.t = 0.0  # time
        self.road_poly = []

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose == 1:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )
        self.car = Car(self.world, *self.track[0][1:4])  # self.track[0][1:4] indicates the starting point of the track

        return self.step(None)[0]

    def step(self, action):
        self.crash = False
        if action is not None:
            action = convert_action(action)
            self.car.steer(action[0] - action[1])  # Update movement properties using action input
            self.car.gas(action[2])
            self.car.brake(action[3])

        self.car.step(1.0 / FPS)  # Step the car physics
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)  # Step the world physics
        self.t += 1.0 / FPS

        self.state = self.render("state_pixels")

        step_reward = 0
        done = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track):
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD or self.crash:
                done = True
                step_reward = -100

        if self.show:
            self.render()
        return self.state, step_reward, done, {}

    def render(self, mode="human"):
        assert mode in ["human", "state_pixels", "rgb_array"]
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label(
                "0000",
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x="left",
                anchor_y="center",
                color=(255, 255, 255, 255),
            )
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        # Animate zoom first second:
        zoom = 0.01 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W / 2
            - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 4
            - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)),
        )
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer)

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == "rgb_array":
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == "state_pixels":
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (
                    win.context._nscontext.view().backingScaleFactor()
                )  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        # self.render_indicators(VIDEO_H, VIDEO_H)
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()

        if mode == "human":
            win.flip()
            return self.viewer.isopen

        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        colors = [0.1, 0.8, 0.1, 1.0] * 4
        polygons_ = [
            +PLAYFIELD,
            +PLAYFIELD,
            0,
            +PLAYFIELD,
            -PLAYFIELD,
            0,
            -PLAYFIELD,
            -PLAYFIELD,
            0,
            -PLAYFIELD,
            +PLAYFIELD,
            0,
        ]

        k = PLAYFIELD / 20.0
        colors.extend([0.1, 0.82, 0.1, 1.0] * 4 * 20 * 20)
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                polygons_.extend(
                    [
                        k * x + k,
                        k * y + 0,
                        0,
                        k * x + 0,
                        k * y + 0,
                        0,
                        k * x + 0,
                        k * y + k,
                        0,
                        k * x + k,
                        k * y + k,
                        0,
                    ]
                )

        for poly, color in self.road_poly:
            colors.extend([color[0], color[1], color[2], 1] * len(poly))
            for p in poly:
                polygons_.extend([p[0], p[1], 0])

        vl = pyglet.graphics.vertex_list(
            len(polygons_) // 3, ("v3f", polygons_), ("c4f", colors)  # gl.GL_QUADS,
        )
        vl.draw(gl.GL_QUADS)

        vl.delete()

    def render_indicators(self, W, H):
        s = W / 40.0
        h = H / 40.0
        colors = [0, 0, 0, 1] * 4
        polygons = [W, 0, 0, W, 5 * h, 0, 0, 5 * h, 0, 0, 0, 0]

        def vertical_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend(
                [
                    place * s,
                    h + h * val,
                    0,
                    (place + 1) * s,
                    h + h * val,
                    0,
                    (place + 1) * s,
                    h,
                    0,
                    (place + 0) * s,
                    h,
                    0,
                ]
            )

        def horiz_ind(place, val, color):
            colors.extend([color[0], color[1], color[2], 1] * 4)
            polygons.extend(
                [
                    (place + 0) * s,
                    4 * h,
                    0,
                    (place + val) * s,
                    4 * h,
                    0,
                    (place + val) * s,
                    2 * h,
                    0,
                    (place + 0) * s,
                    2 * h,
                    0,
                ]
            )

        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )

        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))
        vertical_ind(7, 0.01 * self.car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        vertical_ind(8, 0.01 * self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01 * self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01 * self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        vl = pyglet.graphics.vertex_list(
            len(polygons) // 3, ("v3f", polygons), ("c4f", colors)  # gl.GL_QUADS,
        )
        vl.draw(gl.GL_QUADS)
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()


def make_env():
    env = CarRacing()
    env.show = True
    env = SkipAndStep(env, skip=6)
    env = ProcessFrame(env)
    env = BufferWrapper(env, 6)
    env = ImageToPyTorch(env)
    env = ScaledFloatFrame(env)
    return env


class Agent:
    def __init__(self, env):
        self.env = env
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):

        done_reward = None
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            with torch.set_grad_enabled(False):
                q_vals_v = net(state_v).detach()
            _, act_v = torch.max(q_vals_v, dim=1)
            action = act_v.item()

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


if __name__ == "__main__":
    mode = "agent"  # 'human' or 'agent'

    from pyglet.window import key

    a = np.array([0, 0, 0, 0])


    def key_press(k, mod):
        global restart
        if k == 0xFF0D:
            restart = True
        if k == key.LEFT:
            a[0] = 1
        if k == key.RIGHT:
            a[1] = 1
        if k == key.UP:
            a[2] = 1
        if k == key.DOWN:
            a[3] = 1  # set 1.0 for wheels to block to zero rotation


    def key_release(k, mod):
        if k == key.LEFT and a[0] == 1:
            a[0] = 0
        if k == key.RIGHT and a[1] == 1:
            a[1] = 0
        if k == key.UP:
            a[2] = 0
        if k == key.DOWN:
            a[3] = 0


    num_episodes = 5

    if mode == "human":
        env = CarRacing()
        env.show = True
        env.render()
        env.viewer.window.on_key_press = key_press
        env.viewer.window.on_key_release = key_release
    elif mode == "agent":
        env = make_env()
        env.render()
        model = DQN(env.observation_space.shape, env.action_space.n)
        model.load_state_dict(torch.load("best_model.dat"))
        agent = Agent(env)

    env.reset()

    record_video = False
    if record_video:
        from gym.wrappers import Monitor

        env = Monitor(env, "./recordss/video", force=True)
    isopen = True
    while isopen:
        for episode in range(num_episodes):
            restart = False
            while True:
                if mode == "human":
                    s, r, done, info = env.step(convert_action(a, reverse=True))
                    if done:
                        break
                elif mode == "agent":
                    reward = agent.play_step(model)
                    if reward is not None or restart:
                        break
            env.reset()

        break
    env.close()

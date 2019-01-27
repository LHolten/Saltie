import math
from rlbot.utils.game_state_util import GameState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from examples.levi.input_formatter import LeviInputFormatter
from examples.levi.output_formatter import LeviOutputFormatter
import time
import sys
import os


class PythonExample(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        import torch
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this is for separate process imports
        from examples.levi.torch_model import SymmetricModel
        self.Model = SymmetricModel
        self.torch = torch
        self.controller_state = SimpleControllerState()

        self.frame = 0  # frame counter for timed reset
        self.attempt = 0
        self.brain = 0  # bot counter for generation reset
        self.gen = 0

        self.ball_set = ((0, 1000, 0), (0, 0, 0)), ((1000, 0, 1000), (0, 0, 0)), \
                        ((-1000, 0, 1000), (0, 0, 0)), ((-2000, -2000, 1000), (0, 0, 0)), \
                        ((2000, -2000, 1000), (0, 0, 0))

        self.max_frames = 5000
        self.num_attempts = len(self.ball_set)
        self.pop = 10  # population for bot looping

        self.bot_list = [self.Model() for _ in range(self.pop)]  # list of Individual() objects
        self.parent = [0, 1]  # fittest object

        self.mut_rate = 1  # mutation rate
        self.mut_multiplier = 0.8  # decreasing this will make the mutation rate increase and decrease faster
        self.mut_power = 3  # increasing this will make the mutation rate decrease faster
        self.mut_min = 0.01  # mutation rate will not go below this value

        self.distance_to_ball = [math.inf] * self.max_frames  # set high for easy minimum
        self.min_distance_to_ball = [0] * self.num_attempts  # set 0 for easy current total
        self.bot_fitness = [math.inf] * self.pop  # set high for easy minimum

        self.input_formatter = LeviInputFormatter(team, index)
        self.output_formatter = LeviOutputFormatter(index)

    def initialize_agent(self):
        self.reset()  # reset at start

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # NEURAL NET INPUTS
        inputs = self.input_formatter.create_input_array([packet])
        inputs = [self.torch.from_numpy(x).float() for x in inputs]

        my_car = packet.game_cars[self.index]
        distance_to_ball_x = packet.game_ball.physics.location.x - my_car.physics.location.x
        distance_to_ball_y = packet.game_ball.physics.location.y - my_car.physics.location.y
        distance_to_ball_z = packet.game_ball.physics.location.z - my_car.physics.location.z
        self.distance_to_ball[self.frame] = distance_to_ball_x ** 2 + distance_to_ball_y ** 2 + distance_to_ball_z ** 2

        if packet.game_ball.latest_touch.time_seconds > packet.game_info.seconds_elapsed - 0.02:
            self.distance_to_ball[self.frame] = 0

        # RENDER RESULTS
        action_display = "GEN: " + str(self.gen + 1) + " | BOT: " + str(self.brain + 1)
        draw_debug(self.renderer, action_display)

        # GAME STATE
        car_state = CarState(boost_amount=100)
        velocity = self.ball_set[self.attempt][1]
        ball_state = BallState(
            Physics(angular_velocity=Vector3(0, 0, 0), velocity=Vector3(velocity[0], velocity[1], velocity[2]),
                    location=Vector3(z=1000)))
        game_state = GameState(ball=ball_state, cars={self.index: car_state})
        self.set_game_state(game_state)

        # NEURAL NET OUTPUTS
        with self.torch.no_grad():
            outputs = self.bot_list[self.brain].forward(*inputs)
        self.controller_state = self.output_formatter.format_model_output(outputs, [packet])[0]

        # KILL
        stop_attempt = self.frame > 50 and \
                       (my_car.physics.location.z < 100 or my_car.physics.location.z > 1950 or
                        my_car.physics.location.x < -4000 or my_car.physics.location.x > 4000 or
                        my_car.physics.location.y > 2000 or my_car.physics.location.y < -5000 or
                        self.distance_to_ball[self.frame] > self.distance_to_ball[self.frame - 1])

        # LOOPS
        self.frame += 1
        if self.frame >= self.max_frames or stop_attempt:
            self.frame = 0

            self.calc_min_fitness()
            self.calc_fitness()
            self.calc_fittest()

            stop_brain = (max(self.parent) != self.brain) and self.brain >= 2

            self.attempt += 1
            if self.attempt >= self.num_attempts or stop_brain:
                self.attempt = 0

                self.min_distance_to_ball = [0] * self.num_attempts  # set 0 for easy current total

                self.brain += 1  # change bot every reset
                if self.brain >= self.pop:
                    self.brain = 0  # reset bots after all have gone

                    self.adaptive_mut_rate()

                    self.print_generation_info()

                    # NE Functions
                    self.selection()
                    self.mutate()

                    self.bot_fitness = [math.inf] * self.pop  # reset bot fitness

                    self.gen += 1

            self.controller_state = SimpleControllerState()  # reset controller
            self.reset()  # reset at start

        return self.controller_state

    def calc_min_fitness(self):
        # CALCULATE MINIMUM DISTANCE TO BALL FOR EACH ATTEMPT
        self.min_distance_to_ball[self.attempt] = min(self.distance_to_ball)
        self.distance_to_ball = [math.inf] * self.max_frames

    def calc_fitness(self):
        # CALCULATE AVERAGE OF MINIMUM DISTANCE TO BALL FOR EACH GENOME
        total = sum(self.min_distance_to_ball)
        # sum1 /= len(self.min_distance_to_ball)

        self.bot_fitness[self.brain] = total

    def print_generation_info(self):
        # PRINT GENERATION INFO
        print("")
        print("     GEN = " + str(self.gen + 1))
        print("-------------------------")
        print("FITTEST = BOT " + str(self.parent[0] + 1))
        print("------FITNESS = " + str(math.sqrt(self.bot_fitness[self.parent[0]] / 5)))
        # print("------WEIGHTS = " + str(self.bot_list[self.fittest]))
        for i in range(len(self.bot_list)):
            print("FITNESS OF BOT " + str(i + 1) + " = " + str(math.sqrt(self.bot_fitness[i] / 5)))
        print("------MUTATION RATE = " + str(self.mut_rate))

    def calc_fittest(self):
        temp = [math.inf] * 2
        for i in range(len(self.bot_list)):
            if self.bot_fitness[i] < temp[1]:
                temp[1] = self.bot_fitness[i]
                self.parent[1] = i
            if temp[1] < temp[0]:
                # swap spots
                temp[1], temp[0] = temp[0], temp[1]
                self.parent[1], self.parent[0] = self.parent[0], self.parent[1]

    def adaptive_mut_rate(self):
        if self.parent[0] >= 2:
            # new better bot!
            self.mut_rate /= self.mut_multiplier
        else:
            # no new best bot :(
            self.mut_rate = self.mut_rate * self.mut_multiplier ** self.mut_power

    def reset(self):
        pos = self.ball_set[self.attempt][0]
        # RESET TRAINING ATTRIBUTES AFTER EACH GENOME
        ball_state = BallState(Physics(location=Vector3(pos[0], pos[1], pos[2])))
        car_state = CarState(jumped=False, double_jumped=False, boost_amount=33,
                             physics=Physics(rotation=Rotator(45, 8, 0), velocity=Vector3(0, 0, 0),
                                             angular_velocity=Vector3(0, 0, 0), location=Vector3(0.0, -4608, 500)))
        game_info_state = GameInfoState(game_speed=1.5)
        game_state = GameState(ball=ball_state, cars={self.index: car_state}, game_info=game_info_state)
        self.set_game_state(game_state)
        time.sleep(0.02)

    def selection(self):
        # COPY FITTEST WEIGHTS TO FIRST TWO BOTS
        fittest_dicts = [self.bot_list[parent_index].state_dict() for parent_index in self.parent]
        # for index in range(2):
        #     self.bot_list[index].load_state_dict(fittest_dicts[index])
        #
        # # CROSSOVER TO CREATE THE OTHER BOTS
        # for bot in self.bot_list[2:]:
        #     for param, param_parent1, param_parent2 in zip(bot.parameters(),
        #                                                    self.bot_list[0].parameters(),
        #                                                    self.bot_list[1].parameters()):
        #         mask = self.torch.rand(param.data.size()) * (1 - min(self.mut_rate * 10, 1))
        #         param.data = (param_parent1.data * (1 - mask) + param_parent2.data * mask).clone()
        for index in range(self.pop):
            self.bot_list[index].load_state_dict(fittest_dicts[index % 2])

    def mutate(self):
        # MUTATE NEW GENOMES
        for bot in self.bot_list[2:]:
            new_genes = self.Model()
            for param, param_new in zip(bot.parameters(), new_genes.parameters()):
                # mask = (self.mut_rate / self.torch.rand(param.data.size())).clamp(0, 1)
                mask = self.torch.rand(param.data.size()) < self.mut_rate
                # param.data = (param.data * (1 - mask) + param_new.data * mask).clone()
                param.data[~mask] += (param_new.data[~mask] * self.mut_rate).clone()
                param.data[mask] = param_new.data[mask].clone()


def draw_debug(renderer, action_display):
    renderer.begin_rendering()
    renderer.draw_string_2d(10, 10, 4, 4, action_display, color=renderer.white())
    renderer.end_rendering()

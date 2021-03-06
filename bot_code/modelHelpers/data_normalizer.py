import tensorflow as tf
from bot_code.conversions.input.normalization_input_formatter import NormalizationInputFormatter


class DataNormalizer:
    normalization_array = None
    boolean = [0.0, 1.0]

    def __init__(self, batch_size, feature_creator=None):
        self.batch_size = batch_size
        self.formatter = NormalizationInputFormatter(0, 0, self.batch_size, feature_creator)

    # game_info + score_info + player_car + ball_data +
    # self.flattenArrays(team_members) + self.flattenArrays(enemies) + boost_info
    def create_object(self):
        return lambda: None

    def get_game_info(self):
        info = self.create_object()
        # Game info
        info.bOverTime = self.boolean
        info.bUnlimitedTime = self.boolean
        info.bRoundActive = self.boolean
        info.bBallHasBeenHit = self.boolean
        info.bMatchEnded = self.boolean

        return info

    def create_3D_point(self, x, y, z):
        point = self.create_object()
        point.X = tf.constant(x)
        point.Y = tf.constant(y)
        point.Z = tf.constant(z)
        return point

    def create_3D_rotation(self, pitch, yaw, roll):
        rotator = self.create_object()
        rotator.Pitch = tf.constant(pitch)
        rotator.Yaw = tf.constant(yaw)
        rotator.Roll = tf.constant(roll)
        return rotator

    def createRotVelAng(self, input_velocity, input_angular):
        with tf.name_scope("Rotation"):
            rotation = self.create_3D_rotation([-16384, 16384],  # Pitch
                                               [-32768, 32768],  # Yaw
                                               [-32768, 32768])  # Roll

        with tf.name_scope("Velocity"):
            velocity = self.create_3D_point(
                [-input_velocity, input_velocity],  # Velocity X
                [-input_velocity, input_velocity],  # Y
                [-input_velocity, input_velocity])  # Z

        with tf.name_scope("AngularVelocity"):
            angular = self.create_3D_point(
                [-input_angular, input_angular],  # Angular velocity X
                [-input_angular, input_angular],  # Y
                [-input_angular, input_angular])  # Z

        return (rotation, velocity, angular)

    def get_location(self):
        return self.create_3D_point(
            [-8300, 8300],  # Location X
            [-11800, 11800],  # Y
            [0, 2000])

    def get_car_info(self):
        car = self.create_object()

        car.Location = self.get_location()

        car.Rotation, car.Velocity, car.AngularVelocity = self.createRotVelAng(2300, 5.5)

        car.bDemolished = self.boolean  # Demolished

        car.bOnGround = self.boolean

        car.bJumped = self.boolean  # Jumped
        car.bSuperSonic = self.boolean # Jumped

        car.bDoubleJumped = self.boolean

        car.Team = self.boolean

        car.Boost = [0.0, 100]

        car.Score = self.get_car_score_info()

        return car

    def get_car_score_info(self):
        score = self.create_object()
        score.Score = [0, 100]
        score.Goals = self.boolean
        score.OwnGoals = self.boolean
        score.Assists = self.boolean
        score.Saves = self.boolean
        score.Shots = self.boolean
        score.Demolitions = self.boolean
        return score

    def get_ball_info(self):
        ball = self.create_object()
        ball.Location = self.create_3D_point(
            [-8300, 8300],  # Location X
            [-11800, 11800],  # Y
            [0, 2000])  # Z

        ball.Rotation, ball.Velocity, ball.AngularVelocity = self.createRotVelAng(6000.0, 6.0)

        with tf.name_scope("BallAccerlation"):
            ball.Acceleration = self.create_3D_point(
                self.boolean,  # Acceleration X
                self.boolean,  # Acceleration Y
                self.boolean)  # Acceleration Z

        ball.LatestTouch = self.create_object()

        with tf.name_scope("HitLocation"):
            ball.LatestTouch.sHitLocation = self.get_location()
        with tf.name_scope("HitNormal"):
            ball.LatestTouch.sHitNormal = ball.Velocity
        return ball

    def get_boost_info(self):
        boost_objects = []
        for i in range(35):
            boost_info = self.create_object()
            with tf.name_scope('BoostLocation'):
                boost_info.Location = self.get_location()
            boost_info.bActive = self.boolean
            boost_info.Timer = [0.0, 10000.0]
            boost_objects.append(boost_info)
        return boost_objects

    def get_normalization_array(self):
        state_object = self.create_object()
        # Game info
        with tf.name_scope("Game_Info"):
            state_object.gameInfo = self.get_game_info()
        # Score info

        # Player car info
        state_object.gamecars = []
        car_info = self.get_car_info()
        for i in range(6):
            state_object.gamecars.append(car_info)

        state_object.numCars = len(state_object.gamecars)

        # Ball info
        with tf.name_scope("Ball_Info"):
            state_object.gameball = self.get_ball_info()

        with tf.name_scope("Boost"):
            state_object.gameBoosts = self.get_boost_info()
        return self.formatter.create_input_array(state_object)

    def apply_normalization(self, input_array):
        if self.normalization_array is None:
            self.normalization_array = self.get_normalization_array()

        min = tf.cast(self.normalization_array[0], tf.float32)
        max = tf.cast(self.normalization_array[1], tf.float32)

        diff = max - min

        # error_prevention = tf.cast(tf.equal(diff, 0.0), tf.float32)
        # diff = diff + error_prevention


        #result = (input_array - min) / diff
        result = input_array / diff
        #result = tf.Print(result, [min], 'min', summarize=16)
        #result = tf.Print(result, [max], 'max', summarize=16)
        #result = tf.Print(result, [input_array[0]], 'inp', summarize=30)
        #result = tf.Print(result, [result[0]], 'out', summarize=16)
        result = tf.check_numerics(result, 'post normalization')
        return result

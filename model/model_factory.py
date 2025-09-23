from .common import *


class ModelFactory():
    # 定义工厂映射为类变量
    factory_map = {
        "sr5": lambda type, ik_type: ModelFactory.create_sr5(type, ik_type),
        "er3": lambda type, ik_type: ModelFactory.create_er3(type, ik_type),
        "er3_v2": lambda type, ik_type: ModelFactory.create_er3_v2(type, ik_type),
        "sr5_v2": lambda type, ik_type: ModelFactory.create_sr5_v2(type, ik_type),
        "er3_plus": lambda type, ik_type: ModelFactory.create_er3_plus(type, ik_type),
        "nerv_er3": lambda type, ik_type: ModelFactory.create_nerv_er3(type, ik_type),
    }
    type_supported = ["casadi", "numpy"]

    @classmethod
    def create(cls, name, type='casadi', ik_type=IkType.IK_STANDARD):
        """
        类方法，无需实例化即可创建机器人模型
        """
        if name not in cls.factory_map.keys():
            print('Model not found')
            return
        return cls.factory_map[name](type, ik_type)

    @staticmethod
    def create_sr5(type, ik_type):
        config = rokae_sr5_config
        if type == 'casadi':
            robot = RobotModelCasadi()
        elif type == 'numpy':
            robot = RobotModelNumpy()
        else:
            print('Model type not found')
            return
        robot.build('mat', ik_type, config)
        return robot

    @staticmethod
    def create_sr5_v2(type, ik_type):
        config = rokae_sr5_config
        if type == 'casadi':
            robot = RobotModelCasadi()
        elif type == 'numpy':
            robot = RobotModelNumpy()
        else:
            print('Model type not found')
            return

        upper_bounds = np.radians([175, 135, 140, 175, 175, 175])
        lower_bounds = np.radians([-175, -135, -170, -175, -175, -175])
        config["lower"] = lower_bounds
        config["upper"] = upper_bounds
        robot.build('mat', ik_type, config)
        return robot

    @staticmethod
    def create_er3(type, ik_type):
        config = rokae_er3_config
        if type == 'casadi':
            robot = RobotModelCasadi()
        elif type == 'numpy':
            robot = RobotModelNumpy()
        else:
            print('Model type not found')
            return
        robot.build('dh', ik_type, config)
        return robot

    @staticmethod
    def create_er3_v2(type, ik_type):
        config = rokae_er3_config_expand
        if type == 'casadi':
            robot = RobotModelCasadi()
        elif type == 'numpy':
            robot = RobotModelNumpy()
        else:
            print('Model type not found')
            return
        robot.build('dh', ik_type, config)
        return robot

    @staticmethod
    def create_er3_plus(type, ik_type):
        config = rokae_er3_config_plus
        if type == 'casadi':
            robot = RobotModelCasadi()
        elif type == 'numpy':
            robot = RobotModelNumpy()
        else:
            print('Model type not found')
            return
        robot.build('dh', ik_type, config)
        return robot

    @staticmethod
    def create_nerv_er3(type, ik_type):
        config = nerv_er3_config
        if type == 'casadi':
            robot = RobotModelCasadi()
        elif type == 'numpy':
            robot = RobotModelNumpy()
        else:
            print('Model type not found')
            return
        robot.build('dh', ik_type, config)
        return robot


# 使用示例（无需创建工厂实例）
if __name__ == '__main__':
    sr5 = ModelFactory.create("sr5")
    er3 = ModelFactory.create("er3", "numpy")

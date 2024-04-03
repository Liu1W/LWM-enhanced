import random

from transformer_world_model.utils import ENTITY_IDS, WITH_MESSAGE


def get_avatar_id(obs):
    return obs[..., -1].max()


def get_entity_id_by_role(parsed_manual, role):
    for e in parsed_manual:
        if e[2] == role:
            return ENTITY_IDS[e[0]]
    return None


def get_position_by_id(obs, id):
    entity_ids = obs.reshape(100, -1).max(0).tolist()
    c = entity_ids.index(id)
    pos = obs.reshape(100, -1)[:, c].tolist().index(id)
    row = pos // 10
    col = pos % 10
    return row, col


def out_of_bounds(x):
    return x[0] < 0 or x[0] >= 10 or x[1] < 0 or x[1] >= 10


def get_distance(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


class Oracle:
    INTENTIONS = ["random", "suicide", "survive", "get_message", "go_to_goal"]
    ACTIONS = [(0, -1, 0), (1, 1, 0), (2, 0, -1), (3, 0, 1), (4, 0, 0)]

    def __init__(self, args):
        self.random = random.Random(args.seed + 52398)

    def act(self, obs, parsed_manual, intention):
        if intention == "random":
            return self.random.choice(range(len(self.ACTIONS)))

        if intention == "survive":
            avatar_id = get_avatar_id(obs)
            a_pos = get_position_by_id(obs, avatar_id)
            enemy_id = get_entity_id_by_role(parsed_manual, "enemy")
            e_pos = get_position_by_id(obs, enemy_id)
            goal_id = get_entity_id_by_role(parsed_manual, "goal")
            g_pos = get_position_by_id(obs, goal_id)
            # if message has been obtained, don't care about hitting goal
            if avatar_id == WITH_MESSAGE.id:
                g_pos = None
            # choose action that takes avatar furthest from the enemy
            return self.get_best_action_for_surviving(a_pos, e_pos, g_pos)

        if intention == "suicide":
            avatar_id = get_avatar_id(obs)
            a_pos = get_position_by_id(obs, avatar_id)
            enemy_id = get_entity_id_by_role(parsed_manual, "enemy")
            e_pos = get_position_by_id(obs, enemy_id)
            return self.get_best_action_for_chasing(a_pos, e_pos)

        if intention == "get_message":
            avatar_id = get_avatar_id(obs)
            # if message has been obtained, act randomly
            if avatar_id == WITH_MESSAGE.id:
                return self.act(obs, parsed_manual, "random")
            a_pos = get_position_by_id(obs, avatar_id)
            message_id = get_entity_id_by_role(parsed_manual, "message")
            t_pos = get_position_by_id(obs, message_id)
            # choose action that takes avatar closest to the goal
            return self.get_best_action_for_chasing(a_pos, t_pos)

        if intention == "go_to_goal":
            avatar_id = get_avatar_id(obs)
            a_pos = get_position_by_id(obs, avatar_id)
            # if message has been obtained, go to goal
            if avatar_id == WITH_MESSAGE.id:
                goal_id = get_entity_id_by_role(parsed_manual, "goal")
                t_pos = get_position_by_id(obs, goal_id)
            # else go to message
            else:
                message_id = get_entity_id_by_role(parsed_manual, "message")
                t_p
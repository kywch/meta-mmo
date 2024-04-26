#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

def update_entity_map(short [:, :] entity_map,
                      short [:, ::1] entity_obs,
                      dict entity_attr,
                      dict const_dict):
    cdef short idx, row, col
    cdef short num_team = 0
    cdef short num_enemy = 0

    entity_map[:] = 0
    for idx in range(entity_obs.shape[0]):
        if entity_obs[idx, entity_attr["id"]] == 0:
            continue
        row, col = entity_obs[idx, entity_attr["row"]], entity_obs[idx, entity_attr["col"]]
        if entity_obs[idx, entity_attr["id"]] < 0:
            entity_map[row, col] = max(entity_obs[idx, entity_attr["npc_type"]], entity_map[row, col])
        if entity_obs[idx, entity_attr["id"]] > 0 and entity_obs[idx, entity_attr["npc_type"]] == 0:
            if entity_obs[idx, entity_attr["id"]] in const_dict["my_team"]:
                entity_map[row, col] = max(const_dict["TEAMMATE_REPR"], entity_map[row, col])
                num_team += 1
            else:
                entity_map[row, col] = max(const_dict["ENEMY_REPR"], entity_map[row, col])
                num_enemy += 1
            if entity_obs[idx, entity_attr["id"]] in const_dict["target_destroy"]:
                entity_map[row, col] = max(const_dict["DESTROY_TARGET_REPR"], entity_map[row, col])
            if entity_obs[idx, entity_attr["id"]] in const_dict["target_protect"]:
                entity_map[row, col] = max(const_dict["PROTECT_TARGET_REPR"], entity_map[row, col])
    return num_team, num_enemy

#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

def update_entity_map(short [:, :] entity_map,
                      short [:, :] entity_obs,
                      dict entity_attr,
                      dict const_dict):
    cdef short idx, row, col
    cdef short num_team = 0
    cdef short num_enemy = 0
    cdef short id_col = entity_attr["id"]
    cdef short row_col = entity_attr["row"]
    cdef short col_col = entity_attr["col"]
    cdef short npc_col = entity_attr["npc_type"]

    entity_map[:] = 0
    for idx in range(entity_obs.shape[0]):
        if entity_obs[idx, id_col] == 0:
            continue

        row, col = entity_obs[idx, row_col], entity_obs[idx, col_col]
        if entity_obs[idx, id_col] < 0:
            entity_map[row, col] = max(entity_obs[idx, npc_col], entity_map[row, col])

        if entity_obs[idx, id_col] > 0 and entity_obs[idx, npc_col] == 0:
            if entity_obs[idx, id_col] in const_dict["my_team"]:
                entity_map[row, col] = max(const_dict["TEAMMATE_REPR"], entity_map[row, col])
                num_team += 1
            else:
                entity_map[row, col] = max(const_dict["ENEMY_REPR"], entity_map[row, col])
                num_enemy += 1

            if entity_obs[idx, id_col] in const_dict["target_destroy"]:
                entity_map[row, col] = max(const_dict["DESTROY_TARGET_REPR"], entity_map[row, col])

            if entity_obs[idx, id_col] in const_dict["target_protect"]:
                entity_map[row, col] = max(const_dict["PROTECT_TARGET_REPR"], entity_map[row, col])

    return num_team, num_enemy

def compute_comm_action(bint can_see_target,
                        short my_health,
                        short [:, :] entity_obs,
                        dict entity_attr,
                        dict const_dict):
    cdef short idx
    cdef short peri_enemy = 0
    cdef short peri_npc = 0
    cdef short id_col = entity_attr["id"]

    my_health = (my_health // 34) + 1  # 1 - 3

    for idx in range(entity_obs.shape[0]):
        if entity_obs[idx, id_col] == 0:
            continue

        if entity_obs[idx, id_col] < 0:
            peri_npc += 1

        if entity_obs[idx, id_col] > 0 and \
           entity_obs[idx, id_col] not in const_dict["my_team"]:
            peri_enemy += 1

    peri_enemy = min((peri_enemy+3)//4, 3)  # 0: no enemy, 1: 1-4, 2: 5-8, 3: 9+
    peri_npc = min((peri_npc+3)//4, 3)  # 0: no npc, 1: 1-4, 2: 5-8, 3: 9+

    return can_see_target << 5 | peri_enemy << 4 | peri_npc << 2 | my_health

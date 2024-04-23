from nmmo.task.base_predicates import TickGE
from nmmo.task.task_spec import TaskSpec, check_task_spec

curriculum = []

# Task for the team battle
curriculum.append(
    TaskSpec(
        eval_fn=TickGE,
        eval_fn_kwargs={"num_tick": 1024},
        reward_to="team",
        tags=["team_battle"],
    )
)


if __name__ == "__main__":
    # Import the custom curriculum
    print("------------------------------------------------------------")
    import team_curriculum  # which is this file

    CURRICULUM = team_curriculum.curriculum
    print("The number of training tasks in the curriculum:", len(CURRICULUM))

    # Check if these task specs are valid in the nmmo environment
    # Invalid tasks will crash your agent training
    print("------------------------------------------------------------")
    print("Checking whether the task specs are valid ...")
    results = check_task_spec(CURRICULUM)
    num_error = 0
    for result in results:
        if result["runnable"] is False:
            print("ERROR: ", result["spec_name"])
            num_error += 1
    assert num_error == 0, "Invalid task specs will crash training. Please fix them."
    print("All training tasks are valid.")

    # The task_spec must be picklable to be used for agent training
    print("------------------------------------------------------------")
    print("Checking if the training tasks are picklable ...")
    CURRICULUM_FILE_PATH = "team_curriculum_with_embedding.pkl"
    with open(CURRICULUM_FILE_PATH, "wb") as f:
        import dill

        dill.dump(CURRICULUM, f)
    print("All training tasks are picklable.")

    # To use the curriculum for agent training, the curriculum, task_spec, should be
    # saved to a file with the embeddings using the task encoder. The task encoder uses
    # a coding LLM to encode the task_spec into a vector.
    print("------------------------------------------------------------")
    print("Generating the task spec with embedding file ...")
    from task_encoder import TaskEncoder

    LLM_CHECKPOINT = "deepseek-ai/deepseek-coder-1.3b-instruct"

    # Get the task embeddings for the training tasks and save to file
    # You need to provide the curriculum file as a module to the task encoder
    with TaskEncoder(LLM_CHECKPOINT, team_curriculum) as task_encoder:
        task_encoder.get_task_embedding(CURRICULUM, save_to_file=CURRICULUM_FILE_PATH)
    print("Done.")

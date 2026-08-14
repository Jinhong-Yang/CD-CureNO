from scripts.run_p1_field import build_tasks


def test_field_task_ids_are_deterministic_and_location_specific() -> None:
    first = build_tasks(
        "20260723-1600",
        "legacy_resfno_exact",
        seeds=[1],
        locations=[0, 1, 50],
        git_sha="abcdef0123456789",
    )
    second = build_tasks(
        "20260723-1600",
        "legacy_resfno_exact",
        seeds=[1],
        locations=[0, 1, 50],
        git_sha="abcdef0123456789",
    )
    assert first == second
    assert [task["location"] for task in first] == [0, 1, 50]
    assert len({task["run_id"] for task in first}) == 3
